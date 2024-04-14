// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#define SDL_MAIN_HANDLED

#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include "../../external/whisper.dll.h"
#include <windows.h>

#ifdef _WIN32
#ifdef min 
#undef min
#endif
#ifdef max 
#undef max
#endif
#endif 

// command-line parameters
struct app_whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
	bool openvino	   = true;
	bool use_cuda	   = false;
	bool use_opencl	   = false;

	std::string openvino_encode_device = "CPU";

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const app_whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, app_whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"   || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
		else if (arg == "-oved" || arg == "--ov-e-device")   { params.openvino_encode_device = argv[++i]; }
		else if (arg == "-nov"  || arg == "--no-openvino")   { params.openvino        = false; }
		else if (arg == "-cuda" || arg == "--use-cuda")      { params.use_cuda        = true; }
		else if (arg == "-ocl"  || arg == "--opencl")        { params.use_opencl      = true; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const app_whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N         [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N            [%-7d] audio step size in milliseconds\n",                params.step_ms);
    //fprintf(stderr, "            --length N          [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N            [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID        [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N      [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    //fprintf(stderr, "  -ac N,    --audio-ctx N       [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N       [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N      [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up          [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate         [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback       [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special     [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context      [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG     [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME       [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME        [%-7s] text output file name\n",                          params.fname_out.c_str());
    //fprintf(stderr, "  -tdrz,    --tinydiarize       [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio        [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    //fprintf(stderr, "  -ng,      --no-gpu            [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
	fprintf(stderr, "  -oved D,  --ov-e-device DNAME [%-7s] the OpenVINO device used for encode inference\n",  params.openvino_encode_device.c_str());
	fprintf(stderr, "  -nov,     --no-openvino       [%-7s] disable OpenVINO\n",                               params.openvino ? "false" : "true");
	fprintf(stderr, "  -cuda,    --use-cuda          [%-7s] use CUDA\n",                                       params.use_cuda ? "false" : "true");
	fprintf(stderr, "  -ocl,     --opencl            [%-7s] use OpenCL\n",                                     params.use_opencl ? "false" : "true");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
	app_whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;
	params.use_gpu		  = params.use_cuda;

    // init audio

    audio_async audio(10000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

	const char whisper_library_name[] = "whisper\\whisper.dll";
	HMODULE hLib = LoadLibraryA(whisper_library_name);
	if (!hLib) {
		char path[1024] = {0};
		size_t length = strlen(whisper_library_name) + 1;
		if (GetModuleFileNameA(NULL, path, (DWORD)(sizeof(path) - length))) {
			char * p = path + strlen(path);
			while (p > path) { if (*p == '\\' || *p == '/') { memcpy(p + 1, whisper_library_name, length); break; } p--; }
			hLib = LoadLibraryA(path);
		}
	}

	if (!hLib) {
		fprintf(stderr, "error: unable to load whisper dll library\n");
		exit(0);
	}

	fn_whisper_alloc_context_t whisper_alloc_context = (fn_whisper_alloc_context_t)GetProcAddress(hLib,"whisper_alloc_context");
	fn_whisper_free_context_t whisper_free_context = (fn_whisper_free_context_t)GetProcAddress(hLib,"whisper_free_context");
	fn_whisper_init_context_t whisper_init_context = (fn_whisper_init_context_t)GetProcAddress(hLib,"whisper_init_context");
	fn_whisper_context_process_t whisper_context_process = (fn_whisper_context_process_t)GetProcAddress(hLib,"whisper_context_process");

	p_whisper_dll_context_t ctx = nullptr;
	if (0 != whisper_alloc_context(&ctx)) {
		fprintf(stderr, "error: failed to allocate whisper context\n");
		return 3;
	}

	_whisper_params_t wparams = {0};
	wparams.size = sizeof(wparams);
	strcpy_s(wparams.language,params.language.c_str());
	wparams.nb_threads = params.n_threads;
	wparams.processing_type = processing_cpu;
	if (params.openvino) wparams.processing_type = (_processing_type_t)(wparams.processing_type | processing_openvino);
	if (params.use_cuda) wparams.processing_type = (_processing_type_t)(wparams.processing_type | processing_cuda);
	if (params.use_opencl) wparams.processing_type = (_processing_type_t)(wparams.processing_type | processing_opencl);

	wparams.translate = false;
	wparams.abort_callback = nullptr;
	wparams.user_data = nullptr;
	wparams.frame_size_ms = params.step_ms;
	wparams.keep_size_ms = use_vad ? 0 : params.keep_ms;
	strcpy_s(wparams.openvino_encode_device,params.openvino_encode_device.c_str());

	if (0 != whisper_init_context(ctx,params.model.c_str(),&wparams)) {
		fprintf(stderr, "error: failed to initialize whisper context\n");
		whisper_free_context(&ctx);
		return 4;
	}

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
#if 0
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }
#endif 
    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {
        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio

        if (!use_vad) {
            while (true) {
                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            //const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));
			const int n_samples_take = 0;
            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
        }

        // run the inference
		{
			p_whisper_text_t text = nullptr;
			if (whisper_context_process(ctx, pcmf32_new.data(), pcmf32_new.size() << 2, &text) != 0) {
				fprintf(stderr, "%s: failed to process audio\n", argv[0]);
				whisper_free_context(&ctx);
				return 10;
			}
			++n_iter;
			if (text) {
				do
				{
					if (!params.no_timestamps) {
						printf("[%s --> %s]  ", to_timestamp(text->start).c_str(), to_timestamp(text->stop).c_str());
					}
					printf("%s\n", text->text);
					fflush(stdout);
					text = (p_whisper_text_t)text->next;
				} while (text);
			}
		}
    }

    audio.pause();

	whisper_free_context(&ctx);
	FreeLibrary(hLib);

    return 0;
}
