/* Four-row interleaving keeps each output row's scalar accumulation order
 * unchanged while exposing independent accumulator chains to the CPU. */
#define coli_fp4_matvec_ref coli_fp4_matvec_serial_ref
#define coli_fp8_matvec_ref coli_fp8_matvec_serial_ref
#include "native_quant.c"
#undef coli_fp4_matvec_ref
#undef coli_fp8_matvec_ref

enum { ROW_TILE = 4 };

static void build_decode_tables_v4(float fp4[16], float fp8[256],
                                   float e8[256]) {
    for (int i = 0; i < 16; i++) fp4[i] = coli_e2m1_decode((uint8_t)i);
    for (int i = 0; i < 256; i++) {
        fp8[i] = coli_e4m3fn_decode((uint8_t)i);
        e8[i] = coli_e8m0_decode((uint8_t)i);
    }
}

int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 1 || weight->block_columns != 32) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t packed_stride = columns / 2, scale_stride = columns / 32;
    if (weight->data_bytes != rows * packed_stride ||
        weight->scale_bytes != rows * scale_stride) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(columns / 128);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
    float fp4[16], fp8_unused[256], e8[256];
    build_decode_tables_v4(fp4, fp8_unused, e8);
    const uint8_t *packed = weight->data, *scales = weight->scales;
    int64_t tiles = (weight->rows + ROW_TILE - 1) / ROW_TILE;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < tiles; tile++) {
        int count = (int)(weight->rows - tile * ROW_TILE);
        if (count > ROW_TILE) count = ROW_TILE;
        size_t row_data[ROW_TILE], row_scale[ROW_TILE];
        float sums[ROW_TILE] = {0};
        for (int lane = 0; lane < count; lane++) {
            size_t row = (size_t)tile * ROW_TILE + (size_t)lane;
            row_data[lane] = row * packed_stride;
            row_scale[lane] = row * scale_stride;
        }
        for (size_t base = 0; base < columns; base += 32) {
            float block_scales[ROW_TILE];
            for (int lane = 0; lane < count; lane++)
                block_scales[lane] = e8[scales[row_scale[lane] + base / 32]];
            for (size_t offset = 0; offset < 32; offset++) {
                size_t column = base + offset;
                float x = activation[column];
                for (int lane = 0; lane < count; lane++) {
                    uint8_t byte = packed[row_data[lane] + column / 2];
                    uint8_t code = column & 1 ? byte >> 4 : byte & 15;
                    sums[lane] += x * fp4[code] * block_scales[lane];
                }
            }
        }
        for (int lane = 0; lane < count; lane++)
            output[(size_t)tile * ROW_TILE + (size_t)lane] = sums[lane];
    }
    free(activation_scales); free(activation); return 0;
}

#ifdef __AVX512F__
#include <immintrin.h>

static inline __m512 fp8_decode_16(const uint8_t *src) {
    __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)src));
    __m512i mag = _mm512_and_si512(v, _mm512_set1_epi32(0x7f));
    __m512i sign = _mm512_slli_epi32(_mm512_and_si512(v, _mm512_set1_epi32(0x80)), 24);
    __m512 normal = _mm512_castsi512_ps(_mm512_add_epi32(
        _mm512_slli_epi32(mag, 20), _mm512_set1_epi32(120 << 23)));
    __m512 denormal = _mm512_mul_ps(_mm512_cvtepi32_ps(mag), _mm512_set1_ps(1.0f/512.0f));
    __mmask16 dm = _mm512_cmpeq_epi32_mask(
        _mm512_srli_epi32(mag, 3), _mm512_setzero_si512());
    __m512 magnitude = _mm512_mask_blend_ps(dm, normal, denormal);
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(magnitude), sign));
}
#endif

int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input) {
    if (!output || !weight || !input ||
        weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        weight->scale_format != COLI_SCALE_UE8M0 ||
        !weight->data || !weight->scales || weight->rows < 1 ||
        weight->columns < 1 || weight->columns % 128 ||
        weight->block_rows != 128 || weight->block_columns != 128) return -1;
    size_t rows = (size_t)weight->rows, columns = (size_t)weight->columns;
    size_t scale_rows = (rows + 127) / 128, scale_columns = columns / 128;
    if (weight->data_bytes != rows * columns ||
        weight->scale_bytes != scale_rows * scale_columns) return -1;
    float *activation = malloc(columns * sizeof(*activation));
    uint8_t *activation_scales = malloc(scale_columns);
    if (!activation || !activation_scales) {
        free(activation_scales); free(activation); return -1;
    }
    if (coli_fp8_activation_qdq_ref(activation, activation_scales,
                                    input, columns, 128) != 0) {
        free(activation_scales); free(activation); return -1;
    }
    float e8[256];
    for (int i = 0; i < 256; i++) e8[i] = coli_e8m0_decode((uint8_t)i);
    const uint8_t *data = weight->data, *scales = weight->scales;
    int64_t tiles = (weight->rows + ROW_TILE - 1) / ROW_TILE;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < tiles; tile++) {
        int count = (int)(weight->rows - tile * ROW_TILE);
        if (count > ROW_TILE) count = ROW_TILE;
        size_t rd[ROW_TILE], sr[ROW_TILE];
        float sums[ROW_TILE] = {0};
        for (int lane = 0; lane < count; lane++) {
            size_t row = (size_t)tile * ROW_TILE + (size_t)lane;
            rd[lane] = row * columns;
            sr[lane] = row / 128;
        }
        for (size_t base = 0; base < columns; base += 128) {
            float bs[ROW_TILE];
            for (int lane = 0; lane < count; lane++)
                bs[lane] = e8[scales[sr[lane] * scale_columns + base / 128]];
#ifdef __AVX512F__
            __m512 acc[ROW_TILE];
            for (int lane = 0; lane < count; lane++)
                acc[lane] = _mm512_setzero_ps();
            for (size_t offset = 0; offset < 128; offset += 16) {
                __m512 a = _mm512_loadu_ps(activation + base + offset);
                for (int lane = 0; lane < count; lane++) {
                    __m512 w = fp8_decode_16(data + rd[lane] + base + offset);
                    __m512 ws = _mm512_mul_ps(w, _mm512_set1_ps(bs[lane]));
                    acc[lane] = _mm512_fmadd_ps(a, ws, acc[lane]);
                }
            }
            for (int lane = 0; lane < count; lane++)
                sums[lane] += _mm512_reduce_add_ps(acc[lane]);
#else
            for (size_t offset = 0; offset < 128; offset++) {
                size_t column = base + offset;
                float x = activation[column];
                for (int lane = 0; lane < count; lane++)
                    sums[lane] += x * coli_e4m3fn_decode(data[rd[lane] + column]) * bs[lane];
            }
#endif
        }
        for (int lane = 0; lane < count; lane++)
            output[(size_t)tile * ROW_TILE + (size_t)lane] = sums[lane];
    }
    free(activation_scales); free(activation); return 0;
}
