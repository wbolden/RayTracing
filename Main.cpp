#include "Renderer.cuh"
#include "Timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glfw3.lib")

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <gl\GL.h>
#include <gl\GLU.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define WIDTH 800
#define HEIGHT 600

#define checkErrors(ans) if(ans != cudaSuccess){printf("CUDA error: %s %s %d\n", cudaGetErrorString(ans), __FILE__, __LINE__); cudaGetLastError();}


Renderer render;
GLFWwindow* window;


GLuint vtex;
cudaGraphicsResource_t cudavres;


void display()
{
	
	glClear(GL_COLOR_BUFFER_BIT);

checkErrors(	cudaGraphicsMapResources(1, &cudavres));
	
	cudaArray_t cudavarray;
checkErrors(	cudaGraphicsSubResourceGetMappedArray(&cudavarray, cudavres, 0, 0));

	cudaResourceDesc cudavarrayresdesc;
	memset(&cudavarrayresdesc, 0, sizeof(cudavarrayresdesc));
	cudavarrayresdesc.resType = cudaResourceTypeArray;
	cudavarrayresdesc.res.array.array = cudavarray;
	

	cudaSurfaceObject_t cudavsurfaceobject;
checkErrors(	cudaCreateSurfaceObject(&cudavsurfaceobject, &cudavarrayresdesc));
	

	render.renderFrame(WIDTH, HEIGHT, cudavsurfaceobject);



checkErrors(	cudaGetLastError());
checkErrors(	cudaDeviceSynchronize());


checkErrors(	cudaDestroySurfaceObject(cudavsurfaceobject));
checkErrors(	cudaGraphicsUnmapResources(1, &cudavres));
checkErrors(	cudaStreamSynchronize(0));




	glBindTexture(GL_TEXTURE_2D,vtex);
	{
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		}
		glEnd();
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	glfwSwapBuffers(window);
	glfwPollEvents();


}

int main(int argc, char** argv) 
{
	render = Renderer();

	

	glfwInit();

	window = glfwCreateWindow(WIDTH, HEIGHT, "Ray Tracing", NULL, NULL);
	glfwMakeContextCurrent(window);
	//glfwSetKeyCallback(window, key_callback);



	//glewExperimental = GL_TRUE;
	glewInit();



	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &vtex);

	glBindTexture(GL_TEXTURE_2D, vtex);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);


	checkErrors(cudaGLSetGLDevice(0));
	checkErrors(cudaGraphicsGLRegisterImage(&cudavres, vtex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));


	render.setResolution(WIDTH, HEIGHT);
	while(true)
	{
		display();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
