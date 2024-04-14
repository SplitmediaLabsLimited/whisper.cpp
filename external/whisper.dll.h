//////////////////////////////////////////////////////
#ifndef SML_WHISPER_H
#define SML_WHISPER_H
//////////////////////////////////////////////////////
#include <stdlib.h>
//////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" {
#endif
//////////////////////////////////////////////////////
struct whisper_dll_context_t;
//////////////////////////////////////////////////////
typedef struct whisper_dll_context_t _whisper_dll_context_t, * p_whisper_dll_context_t;
//////////////////////////////////////////////////////
typedef bool (*whisper_abort_proc)(void * user_data);
//////////////////////////////////////////////////////
typedef enum {
	processing_cpu		= 0x00,
	processing_openvino	= 0x01,
	processing_opencl	= 0x02,
	processing_cuda		= 0x04,
}_processing_type_t,*p_processing_type_t;
//////////////////////////////////////////////////////
typedef struct whisper_params {
	// structure size
	uint16_t			size;
	// language default "en"
	char				language[20];
	// Threads default = 4
	uint8_t				nb_threads;
	// Callback param
	void *				user_data;
	// Abort Callback
	whisper_abort_proc	abort_callback;
	// Size of inout frame in miliseconds default = 3000
	size_t				frame_size_ms;
	// translate into english from other languages default = false
	bool				translate;
	// Number of miliseconds keeped from previous call default = 200
	size_t				keep_size_ms;
	// Processing type default = processing_cpu
	_processing_type_t	processing_type;
	// openvino encode device default = "CPU"
	char				openvino_encode_device[20];
	// Using VAD 
	bool				use_vad;
	// VAD Value threshold
	float				vad_thold;
	// VAD Frequency threshold
	float				freq_thold;
}_whisper_params_t,*p_whisper_params_t;
//////////////////////////////////////////////////////
typedef struct whisper_text {
	// Start timestamp
	int64_t start;
	// Stop timestamp
	int64_t stop;
	// Text information
	const char * text;
	// Next info
	const whisper_text * next;
}_whisper_text_t,*p_whisper_text_t;
//////////////////////////////////////////////////////
typedef int (*fn_whisper_alloc_context_t)(p_whisper_dll_context_t * ctx);
typedef int (*fn_whisper_free_context_t)(p_whisper_dll_context_t * ctx);
typedef int (*fn_whisper_init_context_t)(p_whisper_dll_context_t ctx, const char * model_path, p_whisper_params_t params);
typedef int (*fn_whisper_context_reset_t)(p_whisper_dll_context_t ctx);
typedef int (*fn_whisper_get_default_params_t)(p_whisper_params_t params);
typedef int (*fn_whisper_context_get_params_t)(p_whisper_dll_context_t ctx,p_whisper_params_t params);
typedef int (*fn_whisper_context_process_t)(p_whisper_dll_context_t ctx, const void * data, size_t data_size, p_whisper_text_t const * text);
//////////////////////////////////////////////////////
// Allocate context set parameters to default
int whisper_alloc_context(p_whisper_dll_context_t * ctx);
//////////////////////////////////////////////////////
// Release context resources and set it to nullptr
int whisper_free_context(p_whisper_dll_context_t * ctx);
//////////////////////////////////////////////////////
// Initialize whisper context model
int whisper_init_context(p_whisper_dll_context_t ctx, const char * model_path, p_whisper_params_t params);
//////////////////////////////////////////////////////
// Reset context to non initialized state
int whisper_context_reset(p_whisper_dll_context_t ctx);
//////////////////////////////////////////////////////
// Get Default Parameters
int whisper_get_default_params(p_whisper_params_t params);
//////////////////////////////////////////////////////
// Get Current Parameters
int whisper_context_get_params(p_whisper_dll_context_t ctx,p_whisper_params_t params);
//////////////////////////////////////////////////////
// Process media samples
int whisper_context_process(p_whisper_dll_context_t ctx, const void * data, size_t data_size, p_whisper_text_t * text);
//////////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif
//////////////////////////////////////////////////////
#endif
//////////////////////////////////////////////////////