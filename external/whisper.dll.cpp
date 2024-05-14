//////////////////////////////////////////////////////
#define NOMINMAX
#include <windows.h>
#include <minwindef.h>
#include "../whisper.h"
#include "../examples/common.h"
#include "whisper.dll.h"
#include "./base.h"
#include <stdio.h>
#include <thread>
//////////////////////////////////////////////////////
#ifdef __cplusplus
extern "C" {
#endif
//////////////////////////////////////////////////////
struct whisper_dll_context_t {
	// initialization flag
	bool	is_initialized;
	// model path
	char	model_path[512];
	// output text buffer
	struct { 
		void * p;
		size_t size;
	} text_buf;
	// intermediate buffers
	struct {
		float * p;
		size_t size;
		size_t position;
	} buf,keep;
	// Whisper context
	whisper_context * ctx;
	// Language
	char language[20];
	// Threads
	uint8_t	nb_threads;
	// number of frames
	uint32_t nb_frames;
	union {
		// number of frames to keep
		uint32_t nb_keep_frames;
		// number of frames for VAD
		uint32_t nb_vad_frames;
	};
	// Abort Callback
	struct {
		void *	user_data;
		whisper_abort_proc proc;
	} abort_callback;

	bool				translate;
	bool				speed_up;
	bool				append_for_whole_frame;
	int					max_tokens;
	_processing_type_t	processing_type;
	char				openvino_encode_device[50];
	int					best_of;
	int					beam_size;
	int64_t				time_stamp;
	uint64_t			frames_processed;
	uint64_t			frames_total;
	int					audio_ctx;
	bool				use_vad;
	float				vad_thold;
	float				freq_thold;
};
//////////////////////////////////////////////////////
int whisper_alloc_context(p_whisper_dll_context_t * ctx) 
{
	if (!ctx) return -EINVAL;
	*ctx = (p_whisper_dll_context_t)malloc(sizeof(whisper_dll_context_t));
	if (!*ctx) {
		return -ENOMEM;
	}
	memset(*ctx,0x00,sizeof(whisper_dll_context_t));
	const size_t nb_samples = WHISPER_SAMPLE_RATE * 10;
	(*ctx)->keep.p = (float*)malloc(nb_samples * sizeof(float));
	(*ctx)->keep.size = 0;
	(*ctx)->keep.position = 0;
	(*ctx)->buf.p = (*ctx)->keep.p;
	(*ctx)->buf.size = nb_samples;
	(*ctx)->buf.position = 0;
	(*ctx)->append_for_whole_frame = true;

	(*ctx)->text_buf.size = 1024 * 1024;
	(*ctx)->text_buf.p = malloc((*ctx)->text_buf.size);
	
	int ret = whisper_context_reset(*ctx);
	if (ret != 0) {
		whisper_free_context(ctx);
		return ret;
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_get_default_params(p_whisper_params_t params) 
{
	if (!params) return -EINVAL;
	strcpy_s(params->language,"en");
	params->nb_threads = std::thread::hardware_concurrency();
	if (params->nb_threads > 4) params->nb_threads = 4;
	params->frame_size_ms = 3000;
	params->keep_size_ms = 200;
	params->user_data = nullptr;
	params->abort_callback = nullptr;
	params->translate = false;
	params->processing_type = (_processing_type_t)(processing_openvino);
	strcpy_s(params->openvino_encode_device,"CPU");
	if (RTL_CONTAINS_FIELD(params,params->size,use_vad)) {
		params->use_vad = false;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,vad_thold)) {
		params->vad_thold = 0.6f;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,freq_thold)) {
		params->freq_thold = 100.0f;
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_context_set_params(p_whisper_dll_context_t ctx, p_whisper_params_t params)
{
	if (!ctx || !params) return -EINVAL;
	if (ctx->is_initialized) return -ENOENT;
	if (strlen(params->language)) {
		strcpy_s(ctx->language,params->language);
	}
	if (strlen(params->openvino_encode_device)) {
		strcpy_s(ctx->openvino_encode_device,params->openvino_encode_device);
	}
	if (params->nb_threads) {
		ctx->nb_threads = params->nb_threads;
	}
	ctx->abort_callback.proc = params->abort_callback;
	ctx->abort_callback.user_data = params->user_data;
	ctx->translate = params->translate;
	ctx->processing_type = params->processing_type;
	size_t frame_size_ms = params->frame_size_ms;
	size_t keep_size_ms = params->keep_size_ms;
	if (frame_size_ms == 0 || frame_size_ms > 50000) {
		_whisper_params_t _default = {0};
		whisper_get_default_params(&_default);
		frame_size_ms =_default.frame_size_ms;
	}
	if (keep_size_ms) {
		if (keep_size_ms > frame_size_ms) {
			keep_size_ms = frame_size_ms;
		}
	}
	{
		ctx->nb_keep_frames = (uint32_t)((keep_size_ms * WHISPER_SAMPLE_RATE) / 1000);
		ctx->nb_frames = (uint32_t)((frame_size_ms * WHISPER_SAMPLE_RATE) / 1000);
		if (ctx->nb_frames > ctx->buf.size || ctx->nb_keep_frames > ctx->keep.size) {
			ctx->keep.p = (float*)realloc(ctx->keep.p, (ctx->nb_frames + ctx->nb_keep_frames) * sizeof(float));
		}
		ctx->buf.p = ctx->keep.p + ctx->nb_keep_frames;
		ctx->buf.position = 0;
		ctx->keep.position = 0;
		ctx->buf.size = ctx->nb_frames;
		ctx->keep.size = ctx->nb_keep_frames;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,use_vad)) {
		ctx->use_vad = params->use_vad;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,vad_thold)) {
		ctx->vad_thold = params->vad_thold;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,freq_thold)) {
		ctx->freq_thold = params->freq_thold;
	}
	//->
	
	ctx->audio_ctx;
	ctx->best_of;
	ctx->beam_size;
	ctx->speed_up;
	ctx->append_for_whole_frame;
	ctx->max_tokens;
	return 0;
}
//////////////////////////////////////////////////////
int whisper_context_get_params(p_whisper_dll_context_t ctx, p_whisper_params_t params)
{
	if (!ctx || !params) return -EINVAL;
	
	strcpy_s(params->language, ctx->language);
	params->nb_threads = ctx->nb_threads;
	params->abort_callback = ctx->abort_callback.proc;
	params->user_data = ctx->abort_callback.user_data;
	params->translate = ctx->translate;
	params->processing_type = ctx->processing_type;
	params->frame_size_ms = ctx->nb_frames * 1000 / WHISPER_SAMPLE_RATE;
	params->keep_size_ms = ctx->nb_keep_frames * 1000 / WHISPER_SAMPLE_RATE;
	if (RTL_CONTAINS_FIELD(params,params->size,use_vad)) {
		params->use_vad = ctx->use_vad;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,vad_thold)) {
		params->vad_thold = ctx->vad_thold;
	}
	if (RTL_CONTAINS_FIELD(params,params->size,freq_thold)) {
		params->freq_thold = ctx->freq_thold;
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_context_reset(p_whisper_dll_context_t ctx)
{
	if (!ctx) return -EINVAL;
	ctx->buf.position = 0;
	ctx->keep.position = 0;

	ctx->translate = false;
	ctx->nb_frames = (WHISPER_SAMPLE_RATE << 4);
	ctx->nb_keep_frames = 0;
	ctx->best_of = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
	ctx->beam_size = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;
	ctx->processing_type = (_processing_type_t)(processing_openvino | processing_cuda);
	strcpy_s(ctx->openvino_encode_device,"CPU");
	ctx->time_stamp = 0;
	ctx->audio_ctx = 0;
	ctx->speed_up = false;
	ctx->max_tokens = 0;
	ctx->vad_thold = 0.6f;
	ctx->freq_thold = 100.0f;
	ctx->use_vad = false;
	ctx->frames_processed = 0;
	ctx->frames_total = 0;

	_whisper_params_t params = {0};
	params.size = sizeof(params);
	if (0 == whisper_get_default_params(&params)) {
		whisper_context_set_params(ctx, &params);
	}
	if (ctx->is_initialized) {

		if (ctx->ctx) {
			whisper_free(ctx->ctx);
			ctx->ctx = nullptr;
		}
		ctx->is_initialized = false;
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_free_context(p_whisper_dll_context_t * ctx)
{
	if (!ctx) return EINVAL;
	if (*ctx) {
		if ((*ctx)->keep.p) {
			free((*ctx)->keep.p);
		}
		if ((*ctx)->text_buf.p) {
			free((*ctx)->text_buf.p);
		}
		if ((*ctx)->ctx) {
			whisper_free((*ctx)->ctx);
		}
		free(*ctx);
		*ctx = nullptr;
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_init_context(p_whisper_dll_context_t ctx, const char * model_path, p_whisper_params_t params)
{
	if (!ctx) return -EINVAL;
	if (!model_path || !strlen(model_path)) return -ENFILE;
	errno_t ret = strncpy_s(ctx->model_path,model_path,_countof(ctx->model_path) - 1);
	if (ret) return ret;
	if (params) {
		int ret = whisper_context_set_params(ctx,params);
		if (ret != 0) return ret;
	}
	if (_stricmp(ctx->language, "auto") != 0 && whisper_lang_id(ctx->language) == -1) {
		fprintf(stderr, "error: unknown language '%s'\n", ctx->language);
		strcpy_s(ctx->language,"auto");
	}

	whisper_context_params cparams = whisper_context_default_params();
	cparams.use_gpu = ((ctx->processing_type & processing_cuda) != 0);
	s_ggmlBackendType = GGML_BACKEND_CPU;
	if ((ctx->processing_type & processing_cuda) != 0) {
		s_ggmlBackendType |= GGML_BACKEND_CUBLAST;
	}
	if ((ctx->processing_type & processing_opencl) != 0) {
		s_ggmlBackendType |= GGML_BACKEND_CLBLAST;
	}
	ctx->ctx = whisper_init_from_file_with_params(ctx->model_path,cparams);
	if (!ctx->ctx) {
		return -EFAULT;
	}
	if ((ctx->processing_type & processing_openvino) != 0) {
		// initialize openvino encoder. this has no effect on whisper.cpp builds that don't have OpenVINO configured
		whisper_ctx_init_openvino_encoder(ctx->ctx, nullptr, ctx->openvino_encode_device, nullptr);
	}
	ctx->is_initialized = true;
	ctx->frames_processed = 0;
	ctx->frames_total = 0;
	if (!whisper_is_multilingual(ctx->ctx)) {
		if (_stricmp(ctx->language,"en") != 0 || ctx->translate) {
			strcpy_s(ctx->language,"en");
			ctx->translate = false;
			fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
		}
	}
	if (ctx->use_vad) {
		if (!ctx->nb_vad_frames) {
			ctx->nb_vad_frames = (ctx->nb_frames >> 1);
		}
	}
	return 0;
}
//////////////////////////////////////////////////////
int whisper_context_process_frame(p_whisper_dll_context_t ctx, const float * frame, size_t frame_size, p_whisper_text_t last, size_t * offset)
{
	whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	wparams.print_progress   = false;
	wparams.print_special    = false;
	wparams.print_realtime   = false;
	wparams.print_timestamps = true;
	wparams.translate        = ctx->translate;
	wparams.single_segment   = true;
	wparams.max_tokens       = ctx->max_tokens;
	wparams.language         = ctx->language;
	wparams.detect_language	 = (_stricmp(ctx->language,"auto") == 0);
	wparams.n_threads        = (int)ctx->nb_threads;

	wparams.audio_ctx        = 0;
	wparams.speed_up         = ctx->speed_up;
	wparams.tdrz_enable      = true;
	wparams.prompt_tokens	 = nullptr;
	wparams.prompt_n_tokens	 = 0;
	wparams.abort_callback	 = ctx->abort_callback.proc;
	wparams.abort_callback_user_data = ctx->abort_callback.user_data;
	wparams.greedy.best_of        = ctx->best_of;
	wparams.beam_search.beam_size = ctx->beam_size;

	size_t nb_samples = frame_size;
	const float * samples = frame;
	size_t position = ctx->keep.size - ctx->keep.position;
	// Configure input if we have keep buffer
	if (ctx->nb_keep_frames && !ctx->use_vad) {
		if (frame != ctx->buf.p) {
			memcpy(ctx->buf.p,frame,nb_samples * sizeof(float));
		}
		samples = ctx->keep.p + position;
		nb_samples += ctx->keep.position;
	}
	if (whisper_full(ctx->ctx,wparams,samples,(int)nb_samples) != 0) {

		return -ENOSYS;
	}
	// Shift keep buffer data
	if (ctx->nb_keep_frames && !ctx->use_vad) {
		float * p = ctx->keep.p;
		size_t size;
		if (position != 0) {
			p = p + position;
			size = ctx->keep.position + frame_size;
			if (size > ctx->keep.size) {
				p = ctx->keep.p + ((size - ctx->keep.size) % ctx->keep.size ) + frame_size;
				size = ctx->keep.size;
			}
		}
		else {
			p = p + frame_size;
			size = ctx->keep.size;
		}
		memmove(p - frame_size,p, size * sizeof(float));
		ctx->keep.position = (ctx->keep.position + frame_size) % ctx->keep.size;
	}
	ctx->frames_processed += frame_size;
	int64_t start_time = ctx->time_stamp;
	int64_t end_time = (ctx->frames_total + frame_size) * 1000 / WHISPER_SAMPLE_RATE;
	int64_t duration = frame_size * 1000 / WHISPER_SAMPLE_RATE;
	uint8_t * p = (uint8_t *)ctx->text_buf.p;
	p_whisper_text_t prev = nullptr;
	const int n_segments = whisper_full_n_segments(ctx->ctx);
	for (int i = 0; i < n_segments; ++i) {

		const char * text = whisper_full_get_segment_text(ctx->ctx, i);
		if (text) {
			size_t cch = strlen(text);
			if (cch) {
				// TODO: reallocate buffer if needed
				last->start = whisper_full_get_segment_t0(ctx->ctx, i);
				last->stop = whisper_full_get_segment_t1(ctx->ctx, i);
				if (last->start < ctx->time_stamp || last->start > end_time) {
					int64_t temp = last->stop - last->start;
					last->start = ctx->time_stamp;
					last->stop = last->start + temp;
				}
				if (last->stop <= last->start || last->stop - last->start > duration || last->stop > end_time) {
					last->stop = last->start + duration / n_segments;
				}
				ctx->time_stamp = last->stop;
				last->text = (const char *)(p + *offset);
				memcpy((void*)last->text, text, cch + 1);
				*offset += (cch + 1);
				if (prev) {
					prev->next = last;
				}
				prev = last;
				last = (p_whisper_text_t)(((uint8_t*)ctx->text_buf.p) + *offset);
				*offset += sizeof(_whisper_text_t);
				memset(last, 0x00, sizeof(_whisper_text_t));
			}
		}
	}
	ctx->time_stamp = end_time;
	return 0;
}
//////////////////////////////////////////////////////
int whisper_context_process(p_whisper_dll_context_t ctx, const void * data, size_t data_size, p_whisper_text_t * text) 
{
	if (!ctx) return -EINVAL;
	if (!text) return -ENOBUFS;
	if (!ctx->is_initialized) return -ENOENT;
	size_t available_frames = data_size / sizeof(float);
	float * p = (float *)data;
	int ret = 0;
	size_t offset = sizeof(_whisper_text_t);
	p_whisper_text_t first = (p_whisper_text_t)ctx->text_buf.p;
	p_whisper_text_t last = first;
	p_whisper_text_t prev = nullptr;
	memset(last,0x00,sizeof(_whisper_text_t));
	if (p && available_frames) {
		size_t vad_length = ctx->nb_vad_frames;
		if (!vad_length) {
			vad_length = (ctx->nb_frames >> 1);
		}
		std::vector<float> pcm;
		pcm.resize(vad_length);
		int last_ms = (int)((vad_length >> 1) * 1000 / WHISPER_SAMPLE_RATE);
		while (available_frames && ret == 0) {
			size_t length = ctx->nb_frames;
			// we have data from previous 
			if (ctx->buf.position) {
				length = ctx->nb_frames - ctx->buf.position;
				if (length > available_frames) {
					length = available_frames;
				}
				memcpy(ctx->buf.p + ctx->buf.position, p, length * sizeof(float));
				ctx->buf.position += length;
				// Enough data to process
				if (ctx->buf.position >= ctx->nb_frames) {
					if (ctx->use_vad) {
						memcpy(pcm.data(),ctx->buf.p,vad_length * sizeof(float));
						if (::vad_simple(pcm, WHISPER_SAMPLE_RATE, last_ms, ctx->vad_thold, ctx->freq_thold, false)) {
							ret = whisper_context_process_frame(ctx, ctx->buf.p, ctx->buf.position, last, &offset);
							ctx->buf.position = 0;
						}
						else {
							ctx->buf.position -= vad_length;
							memmove(ctx->buf.p,ctx->buf.p + vad_length,ctx->buf.position * sizeof(float));
						}
					}
					else {
						ret = whisper_context_process_frame(ctx, ctx->buf.p, ctx->buf.position, last, &offset);
						ctx->buf.position = 0;
					}
				}
			}
			else {
				// Have enough data for process
				if (available_frames >= length) {
					// use VAD
					if (ctx->use_vad) {

						memcpy(pcm.data(),p,vad_length * sizeof(float));
						if (::vad_simple(pcm, WHISPER_SAMPLE_RATE, last_ms, ctx->vad_thold, ctx->freq_thold, false)) {
							ret = whisper_context_process_frame(ctx, p, length, last, &offset);
						}
						else {
							length = vad_length;
						}
					}
					else {
						ret = whisper_context_process_frame(ctx, p, length, last, &offset);
					}
				}
				else {
					// Skip and append buffer
					length = 0;
				}
			}

			if (ret == 0 && last->text) {
				if (prev) {
					prev->next = last;
				}
				else {
					prev = last;
				}
				while (prev->next) {
					prev = (p_whisper_text_t)prev->next;
				}
				last = (p_whisper_text_t)(((uint8_t*)ctx->text_buf.p) + offset);
				offset += sizeof(_whisper_text_t);
				memset(last, 0x00, sizeof(_whisper_text_t));
			}
			p += length;
			available_frames -= length;
			ctx->frames_total += length;
			if (available_frames < ctx->nb_frames) {
				break;
			}
			if (ctx->abort_callback.proc) {
				if (ctx->abort_callback.proc(ctx->abort_callback.user_data)) {
					ret = -ECANCELED;
					break;
				}
			}
		}
		if (ret != 0) {
			ctx->buf.position = 0;
		}
		else {
			// save outstanding data
			if (available_frames) {
				ctx->frames_total += available_frames;
				if (ctx->buf.position + available_frames > ctx->buf.size) {
					if (available_frames >= ctx->buf.size) {
						p = p + (available_frames - ctx->buf.size);
						available_frames = ctx->buf.size;
						ctx->buf.position = 0;
					}
					else {
						size_t skip = ctx->buf.position + available_frames - ctx->buf.size;
						if (ctx->buf.position > skip) {
							memmove(ctx->buf.p, ctx->buf.p + skip, (ctx->buf.position - skip) * sizeof(float));
						}
						ctx->buf.position = ctx->buf.size - available_frames;
					}
				}
				memcpy(ctx->buf.p + ctx->buf.position, p, available_frames * sizeof(float));
				ctx->buf.position = (ctx->buf.position + available_frames) % ctx->buf.size;
			}
		}
	}
	else {
		// Flush
		if (ctx->buf.position) {
			if (ctx->append_for_whole_frame) {
				const size_t outstanding = ctx->nb_frames - ctx->buf.position;
				memset(ctx->buf.p + ctx->buf.position, 0x00, outstanding * sizeof(float));
				ctx->buf.position = ctx->nb_frames;
			}
			ret = whisper_context_process_frame(ctx, ctx->buf.p, ctx->buf.position, last, &offset);
			ctx->buf.position = 0;
		}
	}
	if (ret == 0 && first->text) {
		*text = first;
	}
	return ret;
}
//////////////////////////////////////////////////////
#ifdef __cplusplus
}
#endif
//////////////////////////////////////////////////////