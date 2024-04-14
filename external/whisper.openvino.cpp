// Wrapper of the OpenVINO Whisper Encoder model
//

#include <windows.h>
#include "openvino/whisper-openvino-encoder.h"

#if __cplusplus
extern "C" {
#endif


static HMODULE s_hLibOpenVINO = nullptr;

// initialize openvino encoder, given path to model xml, device ("CPU", "GPU", etc.), and
// path to cache_dir. Returns null upon failure.
struct whisper_openvino_context * whisper_openvino_init(const char * path_model,
	const char * device,
	const char * cache_dir) {

	struct whisper_openvino_context * ctx = nullptr;

	typedef struct whisper_openvino_context * (__cdecl * fn_whisper_openvino_init)(const char * ,const char * ,const char * );
	
	const char whisper_openvino_library[] = "whisper.openvino.dll";
	if (!s_hLibOpenVINO) {
		char path[1024] = { 0 };
		size_t length = strlen(whisper_openvino_library) + 1;
		HMODULE hModules[] = { NULL,NULL };
		int count = _countof(hModules);
		DWORD Flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
		if (!GetModuleHandleExA(Flags, (LPCSTR)&whisper_openvino_init, &hModules[0])) {
			count--;
			hModules[0] = NULL;
		}
		int idx = 0;
		while (!s_hLibOpenVINO && idx < count) {
			if (GetModuleFileNameA(hModules[idx++], path, (DWORD)(sizeof(path) - length))) {
				char * p = path + strlen(path);
				while (p > path) {  if (*p == '\\' || *p == '/') { p[1] = '\0';SetDllDirectoryA(path); memcpy(p + 1, whisper_openvino_library, length); break; } p--; }
				s_hLibOpenVINO = LoadLibraryA(path);
			}
		}
		if (!s_hLibOpenVINO) {
			s_hLibOpenVINO = LoadLibraryA(whisper_openvino_library);
		}
	}
	if (s_hLibOpenVINO) {
		fn_whisper_openvino_init fn = (fn_whisper_openvino_init)GetProcAddress(s_hLibOpenVINO, "whisper_openvino_init");
		if (fn) {
			ctx = fn(path_model, device, cache_dir);
		}
	}
	return ctx;
}

// clean up a ctx previously returned from whisper_openvino_init()
void whisper_openvino_free(struct whisper_openvino_context * ctx) {

	typedef void (__cdecl * fn_whisper_openvino_free)(struct whisper_openvino_context *);

	if (s_hLibOpenVINO) {
		fn_whisper_openvino_free fn = (fn_whisper_openvino_free)GetProcAddress(s_hLibOpenVINO, "whisper_openvino_free");
		if (fn) {
			fn(ctx);
		}
		FreeLibrary(s_hLibOpenVINO);
		s_hLibOpenVINO = nullptr;
	}
}

// Perform encode using OpenVINO.
// Returns 1 on success
// Returns 0 on failure
int whisper_openvino_encode(
	whisper_openvino_context* ctx,
	ggml_tensor* mel,
	ggml_tensor* out) {

	typedef int (__cdecl * fn_whisper_openvino_encode)(struct whisper_openvino_context *,ggml_tensor* ,ggml_tensor* );

	if (s_hLibOpenVINO) {
		fn_whisper_openvino_encode fn = (fn_whisper_openvino_encode)GetProcAddress(s_hLibOpenVINO, "whisper_openvino_encode");
		if (fn) {
			return fn(ctx, mel, out);
		}
	}
	return 0;
}

#if __cplusplus
}
#endif