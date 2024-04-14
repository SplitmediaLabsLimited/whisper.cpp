
#pragma once 

#ifdef _MSC_VER
#define restrict
#endif
//#define _ALLOW_KEYWORD_MACROS
#define _USE_MATH_DEFINES
#include <math.h>

#define GGML_BACKEND_CPU		0
#define GGML_BACKEND_CUBLAST	1
#define GGML_BACKEND_CLBLAST	2

#define IF_CUBLAS if ((s_ggmlBackendType & GGML_BACKEND_CUBLAST) != 0)
#define IF_CLBLAS if ((s_ggmlBackendType & GGML_BACKEND_CLBLAST) != 0)

typedef struct {
	char function[200];
	void * p;
	int line;
}_mem_t,*p_mem_t;

#ifdef DEBUG
#define ADD_MEM(ptr) if (ptr) { _mem_t t; strcpy_s(t.function,__func__); t.p = ptr; t.line = __LINE__; s_Allocated.push_back(t);}
#define REMOVE_MEM(ptr) if (ptr) { auto it = s_Allocated.begin(); bool found = false; while (it != s_Allocated.end()) { if (it->p == ptr) { s_Allocated.erase(it); found = true; break; } it++; } assert(found); }
#define DUMP_MEM() { auto it = s_Allocated.begin(); while (it != s_Allocated.end()) { fprintf(stderr,"MEMORY: 0x%p Line: %d [%s]\n", it->p, it->line, it->function); it++; } }
#else 
#define ADD_MEM(ptr) 
#define REMOVE_MEM(ptr) 
#define DUMP_MEM() 
#endif 

#ifdef __cplusplus
extern "C" {
#endif

extern int s_ggmlBackendType; /* GGML_BACKEND_CPU */

//extern std::list<_mem_t> s_Allocated;

#ifdef __cplusplus
}
#endif