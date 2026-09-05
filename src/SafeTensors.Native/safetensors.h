/**
 * SafeTensors.NET - C ABI
 *
 * Zero-copy reader for the SafeTensors tensor format.
 *
 * Memory ownership:
 *   - safetensors_get_metadata and safetensors_get_last_error return heap strings that
 *     the caller MUST release with safetensors_free_string. Everything else returns
 *     either a borrowed pointer, valid until safetensors_close, or a status code.
 *   - Pointers from safetensors_get_tensor_data_ptr and the shape pointer inside
 *     safetensors_tensor_info_t belong to the handle. Using either after
 *     safetensors_close reads unmapped memory.
 *
 * Errors:
 *   - Functions returning a pointer report failure as NULL.
 *   - Functions returning int32_t report failure as a negative value.
 *   - safetensors_get_last_error holds the detail and is per-thread.
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(SAFETENSORS_EXPORTS)
    #define SAFETENSORS_API __declspec(dllexport)
  #else
    #define SAFETENSORS_API __declspec(dllimport)
  #endif
#else
  #if defined(SAFETENSORS_EXPORTS)
    #define SAFETENSORS_API __attribute__((visibility("default")))
  #else
    #define SAFETENSORS_API
  #endif
#endif

/**
 * Supported tensor data types.
 */
typedef enum {
    SAFETENSORS_DTYPE_BOOL    = 0,
    SAFETENSORS_DTYPE_U8      = 1,
    SAFETENSORS_DTYPE_I8      = 2,
    SAFETENSORS_DTYPE_I16     = 3,
    SAFETENSORS_DTYPE_U16     = 4,
    SAFETENSORS_DTYPE_F16     = 5,
    SAFETENSORS_DTYPE_BF16    = 6,
    SAFETENSORS_DTYPE_I32     = 7,
    SAFETENSORS_DTYPE_U32     = 8,
    SAFETENSORS_DTYPE_F32     = 9,
    SAFETENSORS_DTYPE_F64     = 10,
    SAFETENSORS_DTYPE_I64     = 11,
    SAFETENSORS_DTYPE_U64     = 12,
    SAFETENSORS_DTYPE_F8_E4M3 = 13,
    SAFETENSORS_DTYPE_F8_E5M2 = 14
} safetensors_dtype_t;

/**
 * Tensor metadata struct.
 */
typedef struct {
    int32_t         dtype;          /**< safetensors_dtype_t */
    int32_t         rank;           /**< Number of dimensions in shape */
    const int64_t*  shape;          /**< Pointer to shape array of size rank */
    uint64_t        byte_length;    /**< Total bytes in raw tensor */
    uint64_t        element_count;  /**< Total number of elements */
} safetensors_tensor_info_t;

/**
 * Opaque handle to an opened SafeTensors archive.
 */
typedef void* safetensors_handle_t;

/**
 * Opens a SafeTensors file in read-only mode using zero-copy memory mapping.
 * @param path UTF-8 path to the .safetensors file.
 * @return Handle to the archive, or NULL on error.
 */
SAFETENSORS_API safetensors_handle_t safetensors_open(const char* path);

/**
 * Closes an opened SafeTensors archive handle and releases all associated memory mappings.
 * @param handle Handle to close.
 */
SAFETENSORS_API void safetensors_close(safetensors_handle_t handle);

/**
 * Gets the total number of tensors stored in the archive.
 * @param handle Archive handle.
 * @return Tensor count, or -1 on error.
 */
SAFETENSORS_API int32_t safetensors_get_tensor_count(safetensors_handle_t handle);

/**
 * Gets the name of the tensor at the specified index.
 * @param handle Archive handle.
 * @param index Index of tensor (0 to count - 1).
 * @param out_buffer Output buffer to write the null-terminated UTF-8 name.
 * @param buffer_size Capacity of out_buffer in bytes.
 * @return Length of the written name in bytes excluding the terminator, or a negative
 *         value on error. If the buffer is too small the return value is the negated
 *         required size including the terminator, so -n means "retry with n bytes".
 */
SAFETENSORS_API int32_t safetensors_get_tensor_name(safetensors_handle_t handle, int32_t index, char* out_buffer, int32_t buffer_size);

/**
 * Retrieves metadata for a tensor by its name.
 * @param handle Archive handle.
 * @param name UTF-8 name of the tensor.
 * @param out_info Pointer to receive tensor information.
 * @return 0 on success, or -1 on error.
 */
SAFETENSORS_API int32_t safetensors_get_tensor_info(safetensors_handle_t handle, const char* name, safetensors_tensor_info_t* out_info);

/**
 * Gets a direct zero-copy pointer to the raw tensor bytes in virtual memory.
 * The pointer remains valid as long as the archive handle is not closed.
 * @param handle Archive handle.
 * @param name UTF-8 name of the tensor.
 * @param out_byte_length Optional pointer to receive total byte length of the tensor.
 * @return Raw pointer to memory-mapped bytes, or NULL on error.
 */
SAFETENSORS_API const void* safetensors_get_tensor_data_ptr(safetensors_handle_t handle, const char* name, uint64_t* out_byte_length);

/**
 * Copies tensor data into a caller-provided destination buffer.
 * @param handle Archive handle.
 * @param name UTF-8 name of the tensor.
 * @param destination Destination buffer.
 * @param destination_size Size of destination buffer in bytes.
 * @return 0 on success, or -1 on error.
 */
SAFETENSORS_API int32_t safetensors_copy_tensor_data(safetensors_handle_t handle, const char* name, void* destination, uint64_t destination_size);

/**
 * Gets a metadata value by key from the __metadata__ dictionary.
 * Caller must free the returned string using safetensors_free_string.
 * @param handle Archive handle.
 * @param key UTF-8 metadata key name.
 * @return Dynamically allocated UTF-8 string, or NULL if not found.
 */
SAFETENSORS_API char* safetensors_get_metadata(safetensors_handle_t handle, const char* key);

/**
 * Gets the last error message recorded on the current thread.
 * Caller must free the returned string using safetensors_free_string.
 * @return Dynamically allocated UTF-8 error string, or NULL if no error.
 */
SAFETENSORS_API char* safetensors_get_last_error(void);

/**
 * Frees a string allocated by safetensors_get_metadata or safetensors_get_last_error.
 * @param str String pointer to free.
 */
SAFETENSORS_API void safetensors_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif /* SAFETENSORS_H */
