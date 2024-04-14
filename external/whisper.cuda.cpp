// Wrapper of the CUDA Whisper.cpp backend
//

#include <windows.h>
#include "ggml-cuda.h"

#if __cplusplus
extern "C" {
#endif

HMODULE LoadBackendCublas() {
	static HMODULE s_hLib = nullptr;
	static bool bLoadCalled = false;
	if (!s_hLib && !bLoadCalled) {
		bLoadCalled = true;
		const char whisper_cuda_library[] = "whisper.cuda.dll";
		char path[1024] = { 0 };
		size_t length = strlen(whisper_cuda_library) + 1;
		HMODULE hModules[] = { NULL,NULL };
		int count = _countof(hModules);
		DWORD Flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
		if (!GetModuleHandleExA(Flags, (LPCSTR)&LoadBackendCublas, &hModules[0])) {
			count--;
			hModules[0] = NULL;
		}
		int idx = 0;
		while (!s_hLib && idx < count) {
			if (GetModuleFileNameA(hModules[idx++], path, (DWORD)(sizeof(path) - length))) {
				char * p = path + strlen(path);
				while (p > path) { if (*p == '\\' || *p == '/') { memcpy(p + 1, whisper_cuda_library, length); break; } p--; }
				s_hLib = LoadLibraryA(path);
			}
		}
		if (!s_hLib) {
			s_hLib = LoadLibraryA(whisper_cuda_library);
		}
	}
	return s_hLib;
}

#if 0
void   ggml_init_cublas(void) {
	typedef void (__cdecl * fn_ggml_init_cublas)();
	static fn_ggml_init_cublas fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_init_cublas)GetProcAddress(hLib, "ggml_init_cublas");
		}
	}
	if (fn) {
		fn();
	}
}

// Returns `true` if there are available CUDA devices and cublas loads successfully; otherwise, it returns `false`.
bool   ggml_cublas_loaded(void) {
	typedef bool (__cdecl * fn_ggml_cublas_loaded)();
	static fn_ggml_cublas_loaded fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cublas_loaded)GetProcAddress(hLib, "ggml_cublas_loaded");
		}
	}
	if (fn) {
		return fn();
	}
	return false;
}

void * ggml_cuda_host_malloc(size_t size) {
	typedef void * (__cdecl * fn_ggml_cuda_host_malloc)(size_t);
	static fn_ggml_cuda_host_malloc fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_host_malloc)GetProcAddress(hLib, "ggml_cuda_host_malloc");
		}
	}
	if (fn) {
		return fn(size);
	}
	return nullptr;
}

void   ggml_cuda_host_free(void * ptr) {
	typedef void * (__cdecl * fn_ggml_cuda_host_free)(void *);
	static fn_ggml_cuda_host_free fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_host_free)GetProcAddress(hLib, "ggml_cuda_host_free");
		}
	}
	if (fn) {
		fn(ptr);
	}
}

bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst)  {
	typedef bool (__cdecl * fn_ggml_cuda_can_mul_mat)(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
	static fn_ggml_cuda_can_mul_mat fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_can_mul_mat)GetProcAddress(hLib, "ggml_cuda_can_mul_mat");
		}
	}
	if (fn) {
		return fn(src0, src1, dst);
	}
	return false;
}

bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
	typedef bool (__cdecl * fn_ggml_cuda_compute_forward)(struct ggml_compute_params *,struct ggml_tensor *);
	static fn_ggml_cuda_compute_forward fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_compute_forward)GetProcAddress(hLib, "ggml_cuda_compute_forward");
		}
	}
	if (fn) {
		return fn(params, tensor);
	}
	return false;
}

int    ggml_cuda_get_device_count(void) {
	typedef int (__cdecl * fn_ggml_cuda_get_device_count)();
	static fn_ggml_cuda_get_device_count fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_get_device_count)GetProcAddress(hLib, "ggml_cuda_get_device_count");
		}
	}
	if (fn) {
		return fn();
	}
	return 0;
}

void   ggml_cuda_get_device_description(int device, char * description, size_t description_size) {
	typedef void * (__cdecl * fn_ggml_cuda_get_device_description)(int device, char * description, size_t description_size);
	static fn_ggml_cuda_get_device_description fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_cuda_get_device_description)GetProcAddress(hLib, "ggml_cuda_get_device_description");
		}
	}
	if (fn) {
		fn(device, description, description_size);
	}
}
#endif
// backend API
ggml_backend_t ggml_backend_cuda_init(int device) {
	typedef ggml_backend_t (__cdecl * fn_ggml_backend_cuda_init)(int);
	static fn_ggml_backend_cuda_init fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_init)GetProcAddress(hLib, "ggml_backend_cuda_init");
		}
	}
	if (fn) {
		return fn(device);
	}
	return nullptr;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
	typedef bool (__cdecl * fn_ggml_backend_is_cuda)(ggml_backend_t);
	static fn_ggml_backend_is_cuda fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_is_cuda)GetProcAddress(hLib, "ggml_backend_is_cuda");
		}
	}
	if (fn) {
		return fn(backend);
	}
	return false;
}

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
	typedef ggml_backend_buffer_type_t (__cdecl * fn_ggml_backend_cuda_buffer_type)(int);
	static fn_ggml_backend_cuda_buffer_type fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_buffer_type)GetProcAddress(hLib, "ggml_backend_cuda_buffer_type");
		}
	}
	if (fn) {
		return fn(device);
	}
	return nullptr;
}
// split tensor buffer that splits matrices by rows across multiple devices
ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split) {
	typedef ggml_backend_buffer_type_t (__cdecl * fn_ggml_backend_cuda_split_buffer_type)(const float *);
	static fn_ggml_backend_cuda_split_buffer_type fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_split_buffer_type)GetProcAddress(hLib, "ggml_backend_cuda_split_buffer_type");
		}
	}
	if (fn) {
		return fn(tensor_split);
	}
	return nullptr;
}
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void)  {
	typedef ggml_backend_buffer_type_t (__cdecl * fn_ggml_backend_cuda_host_buffer_type)();
	static fn_ggml_backend_cuda_host_buffer_type fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_host_buffer_type)GetProcAddress(hLib, "ggml_backend_cuda_host_buffer_type");
		}
	}
	if (fn) {
		return fn();
	}
	return nullptr;
}

int  ggml_backend_cuda_get_device_count(void) {
	typedef int (__cdecl * fn_ggml_backend_cuda_get_device_count)();
	
	static fn_ggml_backend_cuda_get_device_count fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_get_device_count)GetProcAddress(hLib, "ggml_backend_cuda_get_device_count");
		}
	}
	if (fn) {
		return fn();
	}
	return 0;
}
void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
	typedef void (__cdecl * fn_ggml_backend_cuda_get_device_description)(int device, char * description, size_t description_size);
	static fn_ggml_backend_cuda_get_device_description fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_get_device_description)GetProcAddress(hLib, "ggml_backend_cuda_get_device_description");
		}
	}
	if (fn) {
		fn(device, description, description_size);
	}
}
void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
	typedef void (__cdecl * fn_ggml_backend_cuda_get_device_memory)(int , size_t * , size_t *);

	static fn_ggml_backend_cuda_get_device_memory fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_get_device_memory)GetProcAddress(hLib, "ggml_backend_cuda_get_device_memory");
		}
	}
	if (fn) {
		fn(device, free, total);
	}
}

void ggml_backend_cuda_reg_devices(void) {
	typedef void (__cdecl * fn_ggml_backend_cuda_reg_devices)();
	static fn_ggml_backend_cuda_reg_devices fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCublas();
		if (hLib) {
			fn = (fn_ggml_backend_cuda_reg_devices)GetProcAddress(hLib, "ggml_backend_cuda_reg_devices");
		}
	}
	if (fn) {
		fn();
	}
}

#if __cplusplus
}
#endif