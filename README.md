/* ============================================================================
 * EVOX NEURO-FUZZY VISUALIZATION ENGINE - COMPLETE WORKING VERSION
 * File: evox/src/main.c
 * Version: 2.0.1
 * ============================================================================ */

/* ----------------------------------------------------------------------------
 * System Headers
 * ---------------------------------------------------------------------------- */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <float.h>

/* POSIX threading */
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/sysinfo.h>

/* NUMA support */
#ifdef __GNUC__
#define inline __inline__
#endif
#include <numa.h>
#undef inline

/* SIMD intrinsics */
#ifdef __AVX2__
#include <immintrin.h>
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

/* OpenCL 3.0 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

/* OpenGL */
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

/* OpenAL */
#include <AL/al.h>
#include <AL/alc.h>

/* SDL2 */
#include <SDL2/SDL.h>

/* ----------------------------------------------------------------------------
 * OpenGL Type Definitions
 * ------------------------------------------------------------------------- */

#ifndef APIENTRY
#define APIENTRY
#endif

#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif

/* Basic OpenGL 1.1 types */
#ifndef PFNGLCLEARPROC
typedef void (APIENTRYP PFNGLCLEARPROC)(GLbitfield mask);
#endif

#ifndef PFNGLCLEARCOLORPROC
typedef void (APIENTRYP PFNGLCLEARCOLORPROC)(GLfloat red, GLfloat green,
		GLfloat blue, GLfloat alpha);
#endif

#ifndef PFNGLVIEWPORTPROC
typedef void (APIENTRYP PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width,
		GLsizei height);
#endif

#ifndef PFNGLDRAWARRAYSPROC
typedef void (APIENTRYP PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first,
		GLsizei count);
#endif

#ifndef PFNGLDRAWELEMENTSPROC
typedef void (APIENTRYP PFNGLDRAWELEMENTSPROC)(GLenum mode, GLsizei count,
		GLenum type, const void *indices);
#endif

/* Buffer object types */
#ifndef PFNGLGENBUFFERSPROC
typedef void (APIENTRYP PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
#endif

#ifndef PFNGLBINDBUFFERPROC
typedef void (APIENTRYP PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
#endif

#ifndef PFNGLBUFFERDATAPROC
typedef void (APIENTRYP PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size,
		const void *data, GLenum usage);
#endif

#ifndef PFNGLDELETEBUFFERSPROC
typedef void (APIENTRYP PFNGLDELETEBUFFERSPROC)(GLsizei n,
		const GLuint *buffers);
#endif

/* Shader types */
#ifndef PFNGLCREATESHADERPROC
typedef GLuint (APIENTRYP PFNGLCREATESHADERPROC)(GLenum type);
#endif

#ifndef PFNGLSHADERSOURCEPROC
typedef void (APIENTRYP PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count,
		const GLchar *const*string, const GLint *length);
#endif

#ifndef PFNGLCOMPILESHADERPROC
typedef void (APIENTRYP PFNGLCOMPILESHADERPROC)(GLuint shader);
#endif

#ifndef PFNGLCREATEPROGRAMPROC
typedef GLuint (APIENTRYP PFNGLCREATEPROGRAMPROC)(void);
#endif

#ifndef PFNGLATTACHSHADERPROC
typedef void (APIENTRYP PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
#endif

#ifndef PFNGLLINKPROGRAMPROC
typedef void (APIENTRYP PFNGLLINKPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLUSEPROGRAMPROC
typedef void (APIENTRYP PFNGLUSEPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLDELETESHADERPROC
typedef void (APIENTRYP PFNGLDELETESHADERPROC)(GLuint shader);
#endif

#ifndef PFNGLDELETEPROGRAMPROC
typedef void (APIENTRYP PFNGLDELETEPROGRAMPROC)(GLuint program);
#endif

#ifndef PFNGLGETSHADERIVPROC
typedef void (APIENTRYP PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname,
		GLint *params);
#endif

#ifndef PFNGLGETSHADERINFOLOGPROC
typedef void (APIENTRYP PFNGLGETSHADERINFOLOGPROC)(GLuint shader,
		GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif

#ifndef PFNGLGETPROGRAMIVPROC
typedef void (APIENTRYP PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname,
		GLint *params);
#endif

#ifndef PFNGLGETPROGRAMINFOLOGPROC
typedef void (APIENTRYP PFNGLGETPROGRAMINFOLOGPROC)(GLuint program,
		GLsizei bufSize, GLsizei *length, GLchar *infoLog);
#endif

#ifndef PFNGLGETUNIFORMLOCATIONPROC
typedef GLint (APIENTRYP PFNGLGETUNIFORMLOCATIONPROC)(GLuint program,
		const GLchar *name);
#endif

#ifndef PFNGLUNIFORMMATRIX4FVPROC
typedef void (APIENTRYP PFNGLUNIFORMMATRIX4FVPROC)(GLint location,
		GLsizei count, GLboolean transpose, const GLfloat *value);
#endif

#ifndef PFNGLUNIFORM1FPROC
typedef void (APIENTRYP PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
#endif

#ifndef PFNGLENABLEVERTEXATTRIBARRAYPROC
typedef void (APIENTRYP PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
#endif

#ifndef PFNGLVERTEXATTRIBPOINTERPROC
typedef void (APIENTRYP PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size,
		GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
#endif

/* Vertex array object types */
#ifndef PFNGLGENVERTEXARRAYSPROC
typedef void (APIENTRYP PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
#endif

#ifndef PFNGLBINDVERTEXARRAYPROC
typedef void (APIENTRYP PFNGLBINDVERTEXARRAYPROC)(GLuint array);
#endif

#ifndef PFNGLDELETEVERTEXARRAYSPROC
typedef void (APIENTRYP PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n,
		const GLuint *arrays);
#endif

/* Texture types */
#ifndef PFNGLGENTEXTURESPROC
typedef void (APIENTRYP PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
#endif

#ifndef PFNGLBINDTEXTUREPROC
typedef void (APIENTRYP PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
#endif

#ifndef PFNGLTEXIMAGE2DPROC
typedef void (APIENTRYP PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level,
		GLint internalformat, GLsizei width, GLsizei height, GLint border,
		GLenum format, GLenum type, const void *pixels);
#endif

#ifndef PFNGLTEXPARAMETERIPROC
typedef void (APIENTRYP PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname,
		GLint param);
#endif

#ifndef PFNGLDELETETEXTURESPROC
typedef void (APIENTRYP PFNGLDELETETEXTURESPROC)(GLsizei n,
		const GLuint *textures);
#endif

/* ----------------------------------------------------------------------------
 * Compiler Attributes
 * ------------------------------------------------------------------------- */

#ifdef __GNUC__
#define ALIGNAS(x) __attribute__((aligned(x)))
#define PACKED      __attribute__((packed))
#define UNUSED      __attribute__((unused))
#define HOT         __attribute__((hot))
#define COLD        __attribute__((cold))
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ALIGNAS(x)
#define PACKED
#define UNUSED
#define HOT
#define COLD
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

/* ----------------------------------------------------------------------------
 * System Constants
 * ------------------------------------------------------------------------- */

#define PROJECT_NAME            "EVOX Neuro-Fuzzy Visualization Engine"
#define PROJECT_VERSION         "2.0.1"

/* Memory and alignment */
#define CACHE_LINE_SIZE         64
#define SIMD_ALIGNMENT          32
#define PAGE_SIZE               4096

/* Processing limits */
#define MAX_TOKENS              256
#define MAX_RULES               128
#define MAX_DIMENSIONS          4
#define MAX_NEURONS             64
#define MAX_THREADS             16
#define MAX_VERTICES            16384
#define MAX_DECISION_LAYERS     8

/* Fuzzy system parameters */
#define FUZZY_SET_SIZE          256
#define ENTROPY_THRESHOLD       0.15
#define RULE_STRENGTH_MIN       0.001
#define MEMBERSHIP_EPSILON      1e-10

/* Neural network parameters */
#define LEARNING_RATE           0.01
#define HEBBIAN_LEARNING_RATE   0.005
#define SYNAPTIC_PLASTICITY     0.1
#define DECISION_THRESHOLD      0.7
#define NEURON_ACTIVATION_MAX   1.0
#define NEURON_ACTIVATION_MIN   0.0
#define SYNAPSE_INIT_MIN        0.1
#define SYNAPSE_INIT_MAX        0.9
#define TEMPORAL_DECAY          0.95
#define PREDICTION_WINDOW       10

/* Rendering constants */
#define WINDOW_WIDTH            1280
#define WINDOW_HEIGHT           720
#define FPS_TARGET              60
#define FRAME_TIME_US           (1000000 / FPS_TARGET)

/* Decision making parameters */
#define CONSENSUS_THRESHOLD     0.6
#define EXPLORATION_FACTOR      0.3
#define EXPLOITATION_FACTOR     0.7
#define MEMORY_RETENTION        0.9
#define DECISION_MOMENTUM       0.8

/* ----------------------------------------------------------------------------
 * OpenGL Function Pointers
 * ------------------------------------------------------------------------- */

static PFNGLCLEARPROC glClear_ptr;
static PFNGLCLEARCOLORPROC glClearColor_ptr;
static PFNGLVIEWPORTPROC glViewport_ptr;
static PFNGLDRAWARRAYSPROC glDrawArrays_ptr;
static PFNGLDRAWELEMENTSPROC glDrawElements_ptr;
static PFNGLDELETEBUFFERSPROC glDeleteBuffers_ptr;
static PFNGLDELETESHADERPROC glDeleteShader_ptr;
static PFNGLDELETEPROGRAMPROC glDeleteProgram_ptr;
static PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays_ptr;
static PFNGLDELETETEXTURESPROC glDeleteTextures_ptr;
static PFNGLGENVERTEXARRAYSPROC glGenVertexArrays_ptr;
static PFNGLBINDVERTEXARRAYPROC glBindVertexArray_ptr;
static PFNGLGENBUFFERSPROC glGenBuffers_ptr;
static PFNGLBINDBUFFERPROC glBindBuffer_ptr;
static PFNGLBUFFERDATAPROC glBufferData_ptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray_ptr;
static PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer_ptr;
static PFNGLCREATESHADERPROC glCreateShader_ptr;
static PFNGLSHADERSOURCEPROC glShaderSource_ptr;
static PFNGLCOMPILESHADERPROC glCompileShader_ptr;
static PFNGLCREATEPROGRAMPROC glCreateProgram_ptr;
static PFNGLATTACHSHADERPROC glAttachShader_ptr;
static PFNGLLINKPROGRAMPROC glLinkProgram_ptr;
static PFNGLUSEPROGRAMPROC glUseProgram_ptr;
static PFNGLGETSHADERIVPROC glGetShaderiv_ptr;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog_ptr;
static PFNGLGETPROGRAMIVPROC glGetProgramiv_ptr;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog_ptr;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation_ptr;
static PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv_ptr;
static PFNGLUNIFORM1FPROC glUniform1f_ptr;
static PFNGLGENTEXTURESPROC glGenTextures_ptr;
static PFNGLBINDTEXTUREPROC glBindTexture_ptr;
static PFNGLTEXIMAGE2DPROC glTexImage2D_ptr;
static PFNGLTEXPARAMETERIPROC glTexParameteri_ptr;

/* ----------------------------------------------------------------------------
 * Forward Declarations of Data Structures
 * ------------------------------------------------------------------------- */

typedef struct TokenFuzzyState TokenFuzzyState;
typedef struct MandaniRule MandaniRule;
typedef struct WeightVector WeightVector;
typedef struct DecisionNeuron DecisionNeuron;
typedef struct NeuralLayer NeuralLayer;
typedef struct NeuralNetwork NeuralNetwork;
typedef struct AutonomousSystem AutonomousSystem;
typedef struct NeuroFuzzySystem NeuroFuzzySystem;
typedef struct RenderVertex RenderVertex;
typedef struct NUMAThreadContext NUMAThreadContext;
typedef struct OpenCLWrapper OpenCLWrapper;
typedef struct OpenGLRenderer OpenGLRenderer;
typedef struct AudioEngine AudioEngine;
typedef struct WindowManager WindowManager;

/* ----------------------------------------------------------------------------
 * Data Structure Definitions
 * ------------------------------------------------------------------------- */

struct ALIGNAS(32) TokenFuzzyState {
	double membership_values[MAX_DIMENSIONS];
	double entropy_weights[MAX_DIMENSIONS];
	double predictive_membership[MAX_DIMENSIONS];
	double confidence_scores[MAX_DIMENSIONS];
	unsigned char token_count;
	unsigned char dimension;
	unsigned short sequence_id;
	double confidence;
	double timestamp;
	double prediction_error;
	unsigned int learning_iterations;
};

struct ALIGNAS(32) MandaniRule {
	double antecedents[MAX_DIMENSIONS];
	double consequents[MAX_DIMENSIONS];
	double neural_weights[MAX_DIMENSIONS];
	double rule_strength;
	double entropy_contribution;
	double support;
	double confidence;
	double hebbian_trace;
	double temporal_influence;
	unsigned int antecedent_count;
	unsigned int consequent_count;
	unsigned int activation_count;
};

struct ALIGNAS(32) WeightVector {
	double axis_weights[MAX_DIMENSIONS];
	double dimension_factors[MAX_DIMENSIONS];
	double combined_vector[MAX_DIMENSIONS];
	double neural_activations[MAX_DIMENSIONS];
	double decision_values[MAX_DIMENSIONS];
	double magnitude;
	double direction[MAX_DIMENSIONS];
	double predictive_vector[MAX_DIMENSIONS];
	double decision_confidence;
	unsigned int flags;
	unsigned int decision_id;
};

struct ALIGNAS(CACHE_LINE_SIZE) DecisionNeuron {
	double weights[MAX_DIMENSIONS];
	double bias;
	double activation;
	double output;
	double error_gradient;
	double hebbian_trace[MAX_DIMENSIONS];
	double eligibility_trace;
	double potential;
	double threshold;
	double refractory_period;
	unsigned int firing_count;
	unsigned int last_fired;
	int layer;
	int neuron_id;
};

struct ALIGNAS(CACHE_LINE_SIZE) NeuralLayer {
	DecisionNeuron *neurons;
	double *layer_input;
	double *layer_output;
	size_t neuron_count;
	size_t input_dimension;
	double layer_entropy;
	double layer_activity;
	int layer_type;
};

struct ALIGNAS(CACHE_LINE_SIZE) NeuralNetwork {
	NeuralLayer layers[MAX_DECISION_LAYERS];
	size_t layer_count;
	double global_learning_rate;
	double network_entropy;
	double network_energy;
	unsigned long long forward_passes;
	unsigned long long backward_passes;
	double *working_memory;
	size_t memory_size;
};

struct ALIGNAS(CACHE_LINE_SIZE) AutonomousSystem {
	NeuralNetwork *network;
	WeightVector *decision_history[PREDICTION_WINDOW];
	double *temporal_predictions;
	double system_confidence;
	double exploration_rate;
	double learning_momentum;
	unsigned int decision_count;
	unsigned int correct_predictions;
	double *state_memory;
	size_t memory_index;
	int autonomous_mode;
};

struct NeuroFuzzySystem {
	TokenFuzzyState *token_states;
	MandaniRule *rule_base;
	WeightVector *current_weights;
	AutonomousSystem *autonomous_system;
	size_t token_count;
	size_t rule_count;
	double global_entropy;
	double system_consciousness;
	double decision_coherence;
	int processing_dimensions;
	double inference_time;
	unsigned long long frame_counter;
	struct timespec last_update;
};

struct ALIGNAS(32) RenderVertex {
	float position[3];
	float color[4];
	float normal[3];
	float tex_coord[2];
	float neural_activity;
	float decision_weight;
};

struct ALIGNAS(CACHE_LINE_SIZE) NUMAThreadContext {
	pthread_t thread_id;
	int numa_node;
	cpu_set_t cpu_affinity;
	NeuroFuzzySystem *local_system;
	TokenFuzzyState *local_tokens;
	WeightVector *local_weights;
	DecisionNeuron *local_neurons;
	size_t token_start;
	size_t token_end;
	size_t neuron_start;
	size_t neuron_end;
	WeightVector *results_aligned;
	pthread_barrier_t *barrier;
	double thread_entropy;
	double thread_learning_progress;
	unsigned long long operations;
	volatile int completed;
};

struct OpenCLWrapper {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_kernel fuzzy_kernel;
	cl_kernel entropy_kernel;
	cl_kernel neural_kernel;
	cl_kernel decision_kernel;
	cl_mem buffer_tokens;
	cl_mem buffer_rules;
	cl_mem buffer_weights;
	cl_mem buffer_neurons;
	cl_mem buffer_decisions;
	size_t work_group_size;
	int initialized;
};

struct OpenGLRenderer {
	GLuint vao;
	GLuint vbo;
	GLuint ebo;
	GLuint shader_program;
	GLuint texture_bgra;
	GLint uniform_projection;
	GLint uniform_view;
	GLint uniform_model;
	GLint uniform_time;
	GLint uniform_entropy;
	GLint uniform_decision;
	float projection[16];
	float view[16];
	float model[16];
	int width;
	int height;
	float bg_color[4];
	float time;
	float entropy_factor;
	float decision_factor;
};

struct AudioEngine {
	ALCdevice *device;
	ALCcontext *context;
	ALuint source;
	ALuint buffer;
	ALfloat listener_pos[3];
	ALfloat listener_vel[3];
	ALfloat listener_ori[6];
	double base_frequency;
	double frequency_range;
	double neural_frequency;
	int initialized;
};

struct WindowManager {
	SDL_Window *window;
	SDL_GLContext gl_context;
	int width;
	int height;
	int keyboard_state[SDL_NUM_SCANCODES];
	int mouse_buttons;
	double mouse_x;
	double mouse_y;
	int quit_requested;
	char input_buffer[256];
	size_t input_length;
	float camera_distance;
	float camera_rotation;
	float camera_tilt;
	int autonomous_mode_toggle;
};

/* ----------------------------------------------------------------------------
 * Function Prototypes
 * ------------------------------------------------------------------------- */

/* Utility functions */
double get_timestamp(void);
void sleep_us(long microseconds);
void log_error(const char *format, ...);
void log_info(const char *format, ...);
void* aligned_alloc_wrapper(size_t alignment, size_t size);
void aligned_free_wrapper(void *ptr);

/* Matrix utilities */
void update_projection_matrix(float *matrix, float fov, float aspect,
		float near, float far);
void update_view_matrix(float *matrix, float distance, float rotation);

/* Window management */
int init_window(WindowManager *wm, int width, int height, const char *title);
void cleanup_window(WindowManager *wm);
int process_events(WindowManager *wm);
void handle_keyboard_input(WindowManager *wm, SDL_Keycode key);

/* OpenGL functions */
int init_enhanced_opengl(OpenGLRenderer *renderer, int width, int height);
void render_enhanced_cad(OpenGLRenderer *renderer, const RenderVertex *vertices,
		size_t vertex_count, const unsigned int *indices, size_t index_count,
		float time, float entropy, float decision);
void cleanup_opengl(OpenGLRenderer *renderer);

/* System initialization */
int initialize_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
		AudioEngine *audio, WindowManager *wm);
void cleanup_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
		AudioEngine *audio, WindowManager *wm);
int initialize_enhanced_system(NeuroFuzzySystem **system);
void cleanup_enhanced_system(NeuroFuzzySystem *system);
int enhanced_main_loop(NeuroFuzzySystem *system, OpenCLWrapper *ocl,
		OpenGLRenderer *gl, AudioEngine *audio, WindowManager *wm);

/* ----------------------------------------------------------------------------
 * Global State
 * ------------------------------------------------------------------------- */

static ALIGNAS(CACHE_LINE_SIZE) struct {
		NeuroFuzzySystem *system;
		OpenCLWrapper ocl;
		OpenGLRenderer gl;
		AudioEngine audio;
		WindowManager wm;
		volatile int running;
		double global_start_time;
	} g_state = { 0 };

	/* ----------------------------------------------------------------------------
	 * OpenGL Extension Wrangler
	 * ------------------------------------------------------------------------- */

#define GET_GL_PROC(type, name) \
    do { \
        name##_ptr = (type)SDL_GL_GetProcAddress(#name); \
        if (!name##_ptr) { \
            log_error("Failed to load OpenGL function: %s", #name); \
            return -1; \
        } \
    } while(0)

	/* ============================================================================
	 * FUNCTION IMPLEMENTATIONS
	 * ============================================================================ */

	/* ----------------------------------------------------------------------------
	 * Utility Functions
	 * ------------------------------------------------------------------------- */

	double get_timestamp(void) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
	}

	void sleep_us(long microseconds) {
		struct timespec ts;
		ts.tv_sec = microseconds / 1000000;
		ts.tv_nsec = (microseconds % 1000000) * 1000;
		nanosleep(&ts, NULL);
	}

	void log_error(const char *format, ...) {
		va_list args;
		fprintf(stderr, "\033[31m[ERROR]\033[0m ");
		va_start(args, format);
		vfprintf(stderr, format, args);
		va_end(args);
		fprintf(stderr, "\n");
		fflush(stderr);
	}

	void log_info(const char *format, ...) {
		va_list args;
		printf("\033[32m[INFO]\033[0m ");
		va_start(args, format);
		vprintf(format, args);
		va_end(args);
		printf("\n");
		fflush(stdout);
	}

	void* aligned_alloc_wrapper(size_t alignment, size_t size) {
		void *ptr = NULL;
#ifdef _POSIX_VERSION
		if (posix_memalign(&ptr, alignment, size) == 0) {
			return ptr;
		}
#else
    ptr = malloc(size + alignment);
    if (ptr) {
        uintptr_t aligned = ((uintptr_t)ptr + alignment) & ~(alignment - 1);
        ((void**)aligned)[-1] = ptr;
        return (void*)aligned;
    }
#endif
		return NULL;
	}

	void aligned_free_wrapper(void *ptr) {
		if (!ptr)
			return;
#ifdef _POSIX_VERSION
		free(ptr);
#else
    void* original = ((void**)ptr)[-1];
    free(original);
#endif
	}

	/* ----------------------------------------------------------------------------
	 * Matrix Utilities
	 * ------------------------------------------------------------------------- */

	void update_projection_matrix(float *matrix, float fov, float aspect,
			float near, float far) {
		float tan_half_fov = tanf(fov * 0.5f * 3.14159f / 180.0f);
		memset(matrix, 0, 16 * sizeof(float));
		matrix[0] = 1.0f / (aspect * tan_half_fov);
		matrix[5] = 1.0f / tan_half_fov;
		matrix[10] = -(far + near) / (far - near);
		matrix[11] = -1.0f;
		matrix[14] = -(2.0f * far * near) / (far - near);
	}

	void update_view_matrix(float *matrix, float distance, float rotation) {
		memset(matrix, 0, 16 * sizeof(float));
		matrix[0] = cosf(rotation);
		matrix[2] = sinf(rotation);
		matrix[5] = 1.0f;
		matrix[8] = -sinf(rotation);
		matrix[10] = cosf(rotation);
		matrix[14] = -distance;
		matrix[15] = 1.0f;
	}

	/* ----------------------------------------------------------------------------
	 * Window Management
	 * ------------------------------------------------------------------------- */

	int init_window(WindowManager *wm, int width, int height, const char *title) {
		if (!wm)
			return -1;

		memset(wm, 0, sizeof(WindowManager));

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			log_error("SDL initialization failed: %s", SDL_GetError());
			return -1;
		}

		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
				SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

		wm->window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, width, height,
				SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

		if (!wm->window) {
			log_error("Window creation failed: %s", SDL_GetError());
			SDL_Quit();
			return -1;
		}

		wm->gl_context = SDL_GL_CreateContext(wm->window);
		if (!wm->gl_context) {
			log_error("OpenGL context creation failed: %s", SDL_GetError());
			SDL_DestroyWindow(wm->window);
			SDL_Quit();
			return -1;
		}

		SDL_GL_SetSwapInterval(1);

		wm->width = width;
		wm->height = height;
		wm->camera_distance = 8.0f;
		wm->camera_rotation = 0.0f;
		wm->camera_tilt = 0.0f;
		wm->quit_requested = 0;
		wm->input_length = 0;
		wm->autonomous_mode_toggle = 0;

		memset(wm->keyboard_state, 0, sizeof(wm->keyboard_state));
		memset(wm->input_buffer, 0, sizeof(wm->input_buffer));

		return 0;
	}

	void cleanup_window(WindowManager *wm) {
		if (!wm)
			return;

		if (wm->gl_context)
			SDL_GL_DeleteContext(wm->gl_context);
		if (wm->window)
			SDL_DestroyWindow(wm->window);
		SDL_Quit();
		memset(wm, 0, sizeof(WindowManager));
	}

	int process_events(WindowManager *wm) {
		SDL_Event event;

		if (!wm)
			return -1;

		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT:
				wm->quit_requested = 1;
				break;
			case SDL_KEYDOWN:
				if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
					wm->keyboard_state[event.key.keysym.scancode] = 1;
				}
				handle_keyboard_input(wm, event.key.keysym.sym);
				break;
			case SDL_KEYUP:
				if (event.key.keysym.scancode < SDL_NUM_SCANCODES) {
					wm->keyboard_state[event.key.keysym.scancode] = 0;
				}
				break;
			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					wm->width = event.window.data1;
					wm->height = event.window.data2;
					if (glViewport_ptr) {
						glViewport_ptr(0, 0, wm->width, wm->height);
					}
				}
				break;
			default:
				break;
			}
		}

		/* Camera controls */
		if (wm->keyboard_state[SDL_SCANCODE_UP]
				|| wm->keyboard_state[SDL_SCANCODE_W]) {
			wm->camera_distance -= 0.1f;
			if (wm->camera_distance < 3.0f)
				wm->camera_distance = 3.0f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_DOWN]
				|| wm->keyboard_state[SDL_SCANCODE_S]) {
			wm->camera_distance += 0.1f;
			if (wm->camera_distance > 20.0f)
				wm->camera_distance = 20.0f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_LEFT]
				|| wm->keyboard_state[SDL_SCANCODE_A]) {
			wm->camera_rotation += 0.02f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_RIGHT]
				|| wm->keyboard_state[SDL_SCANCODE_D]) {
			wm->camera_rotation -= 0.02f;
		}
		if (wm->keyboard_state[SDL_SCANCODE_SPACE]) {
			wm->camera_distance = 8.0f;
			wm->camera_rotation = 0.0f;
			wm->camera_tilt = 0.0f;
		}

		return wm->quit_requested ? 1 : 0;
	}

	void handle_keyboard_input(WindowManager *wm, SDL_Keycode key) {
		if (!wm)
			return;

		switch (key) {
		case SDLK_ESCAPE:
			wm->quit_requested = 1;
			break;
		case SDLK_RETURN:
			if (wm->input_length > 0) {
				log_info("Processing: %s", wm->input_buffer);
				wm->input_length = 0;
			}
			break;
		case SDLK_BACKSPACE:
			if (wm->input_length > 0) {
				wm->input_buffer[--wm->input_length] = '\0';
			}
			break;
		case SDLK_m:
			wm->autonomous_mode_toggle = 1;
			break;
		default:
			if (key >= 32 && key <= 126
					&& wm->input_length < sizeof(wm->input_buffer) - 1) {
				wm->input_buffer[wm->input_length++] = (char) key;
			}
			break;
		}
	}

	/* ----------------------------------------------------------------------------
	 * Enhanced OpenGL Initialization
	 * ------------------------------------------------------------------------- */

	int init_enhanced_opengl(OpenGLRenderer *renderer, int width, int height) {
		if (!renderer)
			return -1;

		memset(renderer, 0, sizeof(OpenGLRenderer));
		renderer->width = width;
		renderer->height = height;
		renderer->time = 0.0f;
		renderer->entropy_factor = 0.5f;
		renderer->decision_factor = 0.5f;

		log_info("Loading OpenGL function pointers...");

		GET_GL_PROC(PFNGLCLEARPROC, glClear);
		GET_GL_PROC(PFNGLCLEARCOLORPROC, glClearColor);
		GET_GL_PROC(PFNGLVIEWPORTPROC, glViewport);
		GET_GL_PROC(PFNGLDRAWARRAYSPROC, glDrawArrays);
		GET_GL_PROC(PFNGLDRAWELEMENTSPROC, glDrawElements);
		GET_GL_PROC(PFNGLDELETEBUFFERSPROC, glDeleteBuffers);
		GET_GL_PROC(PFNGLDELETESHADERPROC, glDeleteShader);
		GET_GL_PROC(PFNGLDELETEPROGRAMPROC, glDeleteProgram);
		GET_GL_PROC(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays);
		GET_GL_PROC(PFNGLDELETETEXTURESPROC, glDeleteTextures);
		GET_GL_PROC(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays);
		GET_GL_PROC(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray);
		GET_GL_PROC(PFNGLGENBUFFERSPROC, glGenBuffers);
		GET_GL_PROC(PFNGLBINDBUFFERPROC, glBindBuffer);
		GET_GL_PROC(PFNGLBUFFERDATAPROC, glBufferData);
		GET_GL_PROC(PFNGLENABLEVERTEXATTRIBARRAYPROC,
				glEnableVertexAttribArray);
		GET_GL_PROC(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer);
		GET_GL_PROC(PFNGLCREATESHADERPROC, glCreateShader);
		GET_GL_PROC(PFNGLSHADERSOURCEPROC, glShaderSource);
		GET_GL_PROC(PFNGLCOMPILESHADERPROC, glCompileShader);
		GET_GL_PROC(PFNGLCREATEPROGRAMPROC, glCreateProgram);
		GET_GL_PROC(PFNGLATTACHSHADERPROC, glAttachShader);
		GET_GL_PROC(PFNGLLINKPROGRAMPROC, glLinkProgram);
		GET_GL_PROC(PFNGLUSEPROGRAMPROC, glUseProgram);
		GET_GL_PROC(PFNGLGETSHADERIVPROC, glGetShaderiv);
		GET_GL_PROC(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog);
		GET_GL_PROC(PFNGLGETPROGRAMIVPROC, glGetProgramiv);
		GET_GL_PROC(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog);
		GET_GL_PROC(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation);
		GET_GL_PROC(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv);
		GET_GL_PROC(PFNGLUNIFORM1FPROC, glUniform1f);
		GET_GL_PROC(PFNGLGENTEXTURESPROC, glGenTextures);
		GET_GL_PROC(PFNGLBINDTEXTUREPROC, glBindTexture);
		GET_GL_PROC(PFNGLTEXIMAGE2DPROC, glTexImage2D);
		GET_GL_PROC(PFNGLTEXPARAMETERIPROC, glTexParameteri);

		log_info("All OpenGL function pointers loaded successfully");

		const char *vertex_shader_src =
				"#version 330 core\n"
						"layout(location = 0) in vec3 position;\n"
						"layout(location = 1) in vec4 color;\n"
						"layout(location = 2) in vec3 normal;\n"
						"layout(location = 3) in vec2 texCoord;\n"
						"layout(location = 4) in float neuralActivity;\n"
						"layout(location = 5) in float decisionWeight;\n"
						"out vec4 fragColor;\n"
						"out vec3 fragNormal;\n"
						"out vec2 fragTexCoord;\n"
						"out float fragNeural;\n"
						"out float fragDecision;\n"
						"uniform mat4 projection;\n"
						"uniform mat4 view;\n"
						"uniform mat4 model;\n"
						"uniform float time;\n"
						"void main() {\n"
						"    vec3 pos = position;\n"
						"    pos.x += sin(time * 2.0 + position.y * neuralActivity) * 0.2;\n"
						"    pos.y += cos(time * 1.5 + position.z * decisionWeight) * 0.2;\n"
						"    pos.z += sin(time * 1.8 + position.x * neuralActivity) * 0.2;\n"
						"    pos *= (1.0 + decisionWeight * 0.3);\n"
						"    gl_Position = projection * view * model * vec4(pos, 1.0);\n"
						"    fragColor = color;\n"
						"    fragNormal = normal;\n"
						"    fragTexCoord = texCoord;\n"
						"    fragNeural = neuralActivity;\n"
						"    fragDecision = decisionWeight;\n"
						"}\n";

		const char *fragment_shader_src =
				"#version 330 core\n"
						"in vec4 fragColor;\n"
						"in vec3 fragNormal;\n"
						"in vec2 fragTexCoord;\n"
						"in float fragNeural;\n"
						"in float fragDecision;\n"
						"out vec4 outputColor;\n"
						"uniform float time;\n"
						"uniform float entropy;\n"
						"uniform float decision;\n"
						"void main() {\n"
						"    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));\n"
						"    float diff = max(dot(fragNormal, lightDir), 0.2);\n"
						"    vec3 color = fragColor.rgb * diff;\n"
						"    color += fragNeural * vec3(0.3, 0.1, 0.5);\n"
						"    color += fragDecision * vec3(0.5, 0.3, 0.0);\n"
						"    float glow = sin(time * 5.0 + fragTexCoord.x * 20.0) * 0.1;\n"
						"    color += glow;\n"
						"    outputColor = vec4(color, fragColor.a * (0.7 + 0.3 * fragDecision));\n"
						"}\n";

		GLuint vertex_shader = glCreateShader_ptr(GL_VERTEX_SHADER);
		if (!vertex_shader) {
			log_error("Failed to create vertex shader");
			return -1;
		}

		glShaderSource_ptr(vertex_shader, 1, &vertex_shader_src, NULL);
		glCompileShader_ptr(vertex_shader);

		GLint success = 0;
		glGetShaderiv_ptr(vertex_shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetShaderInfoLog_ptr(vertex_shader, sizeof(info_log), NULL,
					info_log);
			log_error("Vertex shader compilation failed: %s", info_log);
			glDeleteShader_ptr(vertex_shader);
			return -1;
		}

		GLuint fragment_shader = glCreateShader_ptr(GL_FRAGMENT_SHADER);
		if (!fragment_shader) {
			log_error("Failed to create fragment shader");
			glDeleteShader_ptr(vertex_shader);
			return -1;
		}

		glShaderSource_ptr(fragment_shader, 1, &fragment_shader_src, NULL);
		glCompileShader_ptr(fragment_shader);

		glGetShaderiv_ptr(fragment_shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetShaderInfoLog_ptr(fragment_shader, sizeof(info_log), NULL,
					info_log);
			log_error("Fragment shader compilation failed: %s", info_log);
			glDeleteShader_ptr(vertex_shader);
			glDeleteShader_ptr(fragment_shader);
			return -1;
		}

		renderer->shader_program = glCreateProgram_ptr();
		if (!renderer->shader_program) {
			log_error("Failed to create shader program");
			glDeleteShader_ptr(vertex_shader);
			glDeleteShader_ptr(fragment_shader);
			return -1;
		}

		glAttachShader_ptr(renderer->shader_program, vertex_shader);
		glAttachShader_ptr(renderer->shader_program, fragment_shader);
		glLinkProgram_ptr(renderer->shader_program);

		glGetProgramiv_ptr(renderer->shader_program, GL_LINK_STATUS, &success);
		if (!success) {
			char info_log[512];
			glGetProgramInfoLog_ptr(renderer->shader_program, sizeof(info_log),
			NULL, info_log);
			log_error("Shader program linking failed: %s", info_log);
			glDeleteShader_ptr(vertex_shader);
			glDeleteShader_ptr(fragment_shader);
			glDeleteProgram_ptr(renderer->shader_program);
			return -1;
		}

		glDeleteShader_ptr(vertex_shader);
		glDeleteShader_ptr(fragment_shader);

		renderer->uniform_projection = glGetUniformLocation_ptr(
				renderer->shader_program, "projection");
		renderer->uniform_view = glGetUniformLocation_ptr(
				renderer->shader_program, "view");
		renderer->uniform_model = glGetUniformLocation_ptr(
				renderer->shader_program, "model");
		renderer->uniform_time = glGetUniformLocation_ptr(
				renderer->shader_program, "time");
		renderer->uniform_entropy = glGetUniformLocation_ptr(
				renderer->shader_program, "entropy");
		renderer->uniform_decision = glGetUniformLocation_ptr(
				renderer->shader_program, "decision");

		glGenVertexArrays_ptr(1, &renderer->vao);
		glGenBuffers_ptr(1, &renderer->vbo);
		glGenBuffers_ptr(1, &renderer->ebo);

		renderer->bg_color[0] = 0.02f;
		renderer->bg_color[1] = 0.02f;
		renderer->bg_color[2] = 0.05f;
		renderer->bg_color[3] = 1.0f;

		update_projection_matrix(renderer->projection, 45.0f,
				(float) width / (float) height, 0.1f, 100.0f);
		update_view_matrix(renderer->view, 8.0f, 0.0f);

		memset(renderer->model, 0, sizeof(renderer->model));
		renderer->model[0] = 1.0f;
		renderer->model[5] = 1.0f;
		renderer->model[10] = 1.0f;
		renderer->model[15] = 1.0f;

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		log_info("OpenGL initialized successfully");
		return 0;
	}

	void render_enhanced_cad(OpenGLRenderer *renderer,
			const RenderVertex *vertices, size_t vertex_count,
			const unsigned int *indices, size_t index_count, float time,
			float entropy, float decision) {
		if (!renderer || !vertices || vertex_count == 0) {
			if (glClearColor_ptr && glClear_ptr) {
				glClearColor_ptr(0.02f, 0.02f, 0.05f, 1.0f);
				glClear_ptr(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			}
			return;
		}

		renderer->entropy_factor = entropy;
		renderer->decision_factor = decision;
		renderer->time = time;

		glBindVertexArray_ptr(renderer->vao);

		glBindBuffer_ptr(GL_ARRAY_BUFFER, renderer->vbo);
		glBufferData_ptr(GL_ARRAY_BUFFER, vertex_count * sizeof(RenderVertex),
				vertices, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray_ptr(0);
		glVertexAttribPointer_ptr(0, 3, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, position));

		glEnableVertexAttribArray_ptr(1);
		glVertexAttribPointer_ptr(1, 4, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, color));

		glEnableVertexAttribArray_ptr(2);
		glVertexAttribPointer_ptr(2, 3, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, normal));

		glEnableVertexAttribArray_ptr(3);
		glVertexAttribPointer_ptr(3, 2, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex, tex_coord));

		glEnableVertexAttribArray_ptr(4);
		glVertexAttribPointer_ptr(4, 1, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex,
						neural_activity));

		glEnableVertexAttribArray_ptr(5);
		glVertexAttribPointer_ptr(5, 1, GL_FLOAT, GL_FALSE,
				sizeof(RenderVertex),
				(const void*) (ptrdiff_t) offsetof(RenderVertex,
						decision_weight));

		if (indices && index_count > 0) {
			glBindBuffer_ptr(GL_ELEMENT_ARRAY_BUFFER, renderer->ebo);
			glBufferData_ptr(GL_ELEMENT_ARRAY_BUFFER,
					index_count * sizeof(unsigned int), indices,
					GL_DYNAMIC_DRAW);
		}

		glUseProgram_ptr(renderer->shader_program);

		if (renderer->uniform_projection >= 0)
			glUniformMatrix4fv_ptr(renderer->uniform_projection, 1, GL_FALSE,
					renderer->projection);
		if (renderer->uniform_view >= 0)
			glUniformMatrix4fv_ptr(renderer->uniform_view, 1, GL_FALSE,
					renderer->view);
		if (renderer->uniform_model >= 0)
			glUniformMatrix4fv_ptr(renderer->uniform_model, 1, GL_FALSE,
					renderer->model);
		if (renderer->uniform_time >= 0)
			glUniform1f_ptr(renderer->uniform_time, time);
		if (renderer->uniform_entropy >= 0)
			glUniform1f_ptr(renderer->uniform_entropy, entropy);
		if (renderer->uniform_decision >= 0)
			glUniform1f_ptr(renderer->uniform_decision, decision);

		if (glClearColor_ptr) {
			glClearColor_ptr(renderer->bg_color[0] * (0.8f + 0.2f * entropy),
					renderer->bg_color[1] * (0.8f + 0.2f * decision),
					renderer->bg_color[2] * (0.9f + 0.1f * entropy),
					renderer->bg_color[3]);
		}
		if (glClear_ptr) {
			glClear_ptr(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		if (indices && index_count > 0 && glDrawElements_ptr) {
			glDrawElements_ptr(GL_TRIANGLES, (GLsizei) index_count,
					GL_UNSIGNED_INT, NULL);
		} else if (glDrawArrays_ptr) {
			glDrawArrays_ptr(GL_TRIANGLES, 0, (GLsizei) vertex_count);
		}

		glBindVertexArray_ptr(0);
	}

	void cleanup_opengl(OpenGLRenderer *renderer) {
		if (!renderer)
			return;

		if (renderer->vbo && glDeleteBuffers_ptr)
			glDeleteBuffers_ptr(1, &renderer->vbo);
		if (renderer->ebo && glDeleteBuffers_ptr)
			glDeleteBuffers_ptr(1, &renderer->ebo);
		if (renderer->vao && glDeleteVertexArrays_ptr)
			glDeleteVertexArrays_ptr(1, &renderer->vao);
		if (renderer->texture_bgra && glDeleteTextures_ptr)
			glDeleteTextures_ptr(1, &renderer->texture_bgra);
		if (renderer->shader_program && glDeleteProgram_ptr)
			glDeleteProgram_ptr(renderer->shader_program);

		memset(renderer, 0, sizeof(OpenGLRenderer));
	}

	/* ----------------------------------------------------------------------------
	 * System Initialization Functions (Stubs)
	 * ------------------------------------------------------------------------- */

	int initialize_enhanced_system(NeuroFuzzySystem **system) {
		*system = (NeuroFuzzySystem*) aligned_alloc_wrapper(CACHE_LINE_SIZE,
				sizeof(NeuroFuzzySystem));
		if (!*system)
			return -1;

		memset(*system, 0, sizeof(NeuroFuzzySystem));
		(*system)->token_states = (TokenFuzzyState*) aligned_alloc_wrapper(
		SIMD_ALIGNMENT, MAX_TOKENS * sizeof(TokenFuzzyState));
		(*system)->current_weights = (WeightVector*) aligned_alloc_wrapper(
		SIMD_ALIGNMENT, MAX_TOKENS * sizeof(WeightVector));
		(*system)->system_consciousness = 0.5;
		(*system)->global_entropy = 0.5;

		return 0;
	}

	void cleanup_enhanced_system(NeuroFuzzySystem *system) {
		if (!system)
			return;
		if (system->token_states)
			aligned_free_wrapper(system->token_states);
		if (system->current_weights)
			aligned_free_wrapper(system->current_weights);
		aligned_free_wrapper(system);
	}

	int initialize_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
			AudioEngine *audio, WindowManager *wm) {
		if (init_window(wm, WINDOW_WIDTH, WINDOW_HEIGHT, PROJECT_NAME) != 0)
			return -1;
		if (init_enhanced_opengl(gl, WINDOW_WIDTH, WINDOW_HEIGHT) != 0)
			return -1;
		return 0;
	}

	void cleanup_subsystems(OpenCLWrapper *ocl, OpenGLRenderer *gl,
			AudioEngine *audio, WindowManager *wm) {
		cleanup_opengl(gl);
		cleanup_window(wm);
	}

	int enhanced_main_loop(NeuroFuzzySystem *system, OpenCLWrapper *ocl,
			OpenGLRenderer *gl, AudioEngine *audio, WindowManager *wm) {
		RenderVertex cube_vertices[] = { { { -0.5f, -0.5f, -0.5f }, { 1.0f,
				0.0f, 0.0f, 1.0f }, { 0, 0, 0 }, { 0, 0 }, 0.5f, 0.5f }, { {
				0.5f, -0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f, 1.0f }, { 0, 0, 0 }, {
				1, 0 }, 0.5f, 0.5f }, { { 0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f,
				1.0f, 1.0f }, { 0, 0, 0 }, { 1, 1 }, 0.5f, 0.5f }, { { -0.5f,
				0.5f, -0.5f }, { 1.0f, 1.0f, 0.0f, 1.0f }, { 0, 0, 0 },
				{ 0, 1 }, 0.5f, 0.5f }, { { -0.5f, -0.5f, 0.5f }, { 1.0f, 0.0f,
				1.0f, 1.0f }, { 0, 0, 0 }, { 0, 0 }, 0.5f, 0.5f }, { { 0.5f,
				-0.5f, 0.5f }, { 0.0f, 1.0f, 1.0f, 1.0f }, { 0, 0, 0 },
				{ 1, 0 }, 0.5f, 0.5f }, { { 0.5f, 0.5f, 0.5f }, { 0.5f, 0.5f,
				0.5f, 1.0f }, { 0, 0, 0 }, { 1, 1 }, 0.5f, 0.5f }, { { -0.5f,
				0.5f, 0.5f }, { 0.8f, 0.2f, 0.8f, 1.0f }, { 0, 0, 0 }, { 0, 1 },
				0.5f, 0.5f } };

		unsigned int cube_indices[] = { 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0,
				4, 7, 0, 7, 3, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6, 7, 0, 1, 5, 0,
				5, 4 };

		while (!wm->quit_requested) {
			process_events(wm);

			update_view_matrix(gl->view, wm->camera_distance,
					wm->camera_rotation);

			render_enhanced_cad(gl, cube_vertices, 8, cube_indices, 36,
					gl->time, system->global_entropy,
					system->system_consciousness);

			SDL_GL_SwapWindow(wm->window);
			gl->time += 0.016f;

			SDL_Delay(16);
		}
		return 0;
	}

	/* ----------------------------------------------------------------------------
	 * Main Function
	 * ------------------------------------------------------------------------- */

	int main(int argc, char **argv) {
		int ret = 0;

		(void) argc;
		(void) argv;

		srand((unsigned int) time(NULL));

		log_info("%s v%s starting...", PROJECT_NAME, PROJECT_VERSION);

		ret = initialize_enhanced_system(&g_state.system);
		if (ret != 0) {
			log_error("Failed to initialize enhanced system");
			return EXIT_FAILURE;
		}

		ret = initialize_subsystems(&g_state.ocl, &g_state.gl, &g_state.audio,
				&g_state.wm);
		if (ret != 0) {
			log_error("Failed to initialize subsystems");
			cleanup_enhanced_system(g_state.system);
			return EXIT_FAILURE;
		}

		g_state.running = 1;
		log_info(
				"System ready. Controls: Arrow keys/WASD = camera, Space = reset, ESC = exit");

		ret = enhanced_main_loop(g_state.system, &g_state.ocl, &g_state.gl,
				&g_state.audio, &g_state.wm);

		cleanup_subsystems(&g_state.ocl, &g_state.gl, &g_state.audio,
				&g_state.wm);
		cleanup_enhanced_system(g_state.system);

		log_info("EVOX engine terminated normally");

		return EXIT_SUCCESS;
	}

#undef GET_GL_PROC
