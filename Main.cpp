#include "Display.h"
#include "Renderer.cuh"
#include "Timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glfw3.lib")

#define WIDTH 800
#define HEIGHT 600

int main(int argc, char** argv) 
{
	GLFWwindow* window;

	glfwInit();
	window = glfwCreateWindow(WIDTH, HEIGHT, "Ray Tracing", NULL, NULL);
	glfwMakeContextCurrent(window);

	Display display = Display(window);

	display.initRenderTexture();
	display.createRenderer();
	
	Timer timer = Timer();

	while(true)
	{
		//todo: get inputs

		timer.start();

		display.displayFrame(window);

		timer.stop();
		if(timer.getFrameCount() % 10 == 0)
		{
			printf("%f\n", timer.getFPS());
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
