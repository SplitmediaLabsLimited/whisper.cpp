// Wrapper of the OpenCL Whisper.cpp backend
//

#include <windows.h>
#include "ggml-opencl.h"

#if __cplusplus
extern "C" {
#endif

HMODULE LoadBackendCLBlast() {
	static HMODULE s_hLib = nullptr;
	static bool bLoadCalled = false;
	if (!s_hLib && !bLoadCalled) {
		bLoadCalled = true;
		const char whisper_clblast_library[] = "whisper.clblast.dll";
		char path[1024] = { 0 };
		size_t length = strlen(whisper_clblast_library) + 1;
		HMODULE hModules[] = { NULL,NULL };
		int count = _countof(hModules);
		DWORD Flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
		if (!GetModuleHandleExA(Flags, (LPCSTR)&LoadBackendCLBlast, &hModules[0])) {
			count--;
			hModules[0] = NULL;
		}
		int idx = 0;
		while (!s_hLib && idx < count) {
			if (GetModuleFileNameA(hModules[idx++], path, (DWORD)(sizeof(path) - length))) {
				char * p = path + strlen(path);
				while (p > path) { if (*p == '\\' || *p == '/') { memcpy(p + 1, whisper_clblast_library, length); break; } p--; }
				s_hLib = LoadLibraryA(path);
			}
		}
		if (!s_hLib) {
			s_hLib = LoadLibraryA(whisper_clblast_library);
		}
	}
	return s_hLib;
}

void ggml_cl_init(void) {
	typedef void (__cdecl * fn_ggml_cl_init)();
	static fn_ggml_cl_init fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_init)GetProcAddress(hLib, "ggml_cl_init");
		}
	}
	if (fn) {
		fn();
	}
}

void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
	typedef void (__cdecl * fn_ggml_cl_mul)(const struct ggml_tensor * , const struct ggml_tensor * , struct ggml_tensor * );
	
	static fn_ggml_cl_mul fn = nullptr;

	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_mul)GetProcAddress(hLib, "ggml_cl_mul");
		}
	}
	if (fn) {
		fn(src0, src1, dst);
	}
}

bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
	
	typedef bool (__cdecl * fn_ggml_cl_can_mul_mat)(const struct ggml_tensor * , const struct ggml_tensor * ,const struct ggml_tensor * );
	static fn_ggml_cl_can_mul_mat fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_can_mul_mat)GetProcAddress(hLib, "ggml_cl_can_mul_mat");
		}
	}
	if (fn) {
		return fn(src0, src1, dst);
	}
	return false;
}

size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {

	typedef size_t (__cdecl * fn_ggml_cl_mul_mat_get_wsize)(const struct ggml_tensor * , const struct ggml_tensor * ,struct ggml_tensor * );
	static fn_ggml_cl_mul_mat_get_wsize fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_mul_mat_get_wsize)GetProcAddress(hLib, "ggml_cl_mul_mat_get_wsize");
		}
	}
	if (fn) {
		return fn(src0, src1, dst);
	}
	return 0;
}

void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {

	typedef void (__cdecl * fn_ggml_cl_mul_mat)(const struct ggml_tensor * , const struct ggml_tensor * ,struct ggml_tensor * ,void *, size_t);
	static fn_ggml_cl_mul_mat fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_mul_mat)GetProcAddress(hLib, "ggml_cl_mul_mat");
		}
	}
	if (fn) {
		fn(src0, src1, dst,wdata, wsize);
	}
}

void ggml_cl_free_data(const struct ggml_tensor* tensor) {

	typedef void (__cdecl * fn_ggml_cl_free_data)(const struct ggml_tensor *);
	static fn_ggml_cl_free_data fn = nullptr;

	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_free_data)GetProcAddress(hLib, "ggml_cl_free_data");
		}
	}
	if (fn) {
		fn(tensor);
	}
}

void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor) {

	typedef void (__cdecl * fn_ggml_cl_transform_tensor)(void * ,const struct ggml_tensor *);
	static fn_ggml_cl_transform_tensor fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_transform_tensor)GetProcAddress(hLib, "ggml_cl_transform_tensor");
		}
	}
	if (fn) {
		fn(data,tensor);
	}
}

ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void) {

	typedef ggml_backend_buffer_type_t (__cdecl * fn_ggml_backend_opencl_buffer_type)();
	static fn_ggml_backend_opencl_buffer_type fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_backend_opencl_buffer_type)GetProcAddress(hLib, "ggml_backend_opencl_buffer_type");
		}
	}
	if (fn) {
		return fn();
	}
	return nullptr;
}

void ggml_cl_add(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {

	typedef void (__cdecl * fn_ggml_cl_add)(const struct ggml_tensor *,const struct ggml_tensor *,struct ggml_tensor *);
	static fn_ggml_cl_add fn = nullptr;
	if (!fn) {
		HMODULE hLib = LoadBackendCLBlast();
		if (hLib) {
			fn = (fn_ggml_cl_add)GetProcAddress(hLib, "ggml_cl_add");
		}
	}
	if (fn) {
		fn(src0, src1, dst);
	}
}

#if __cplusplus
}
#endif