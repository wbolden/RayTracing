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

#define WIDTH 600
#define HEIGHT 600

int main(int argc, char** argv) 
{
	GLFWwindow* window;

	glfwInit();
	window = glfwCreateWindow(WIDTH, HEIGHT, "Ray Tracing", NULL, NULL);
	glfwMakeContextCurrent(window);

	Display display = Display(window);

	display.initWindow();
	display.createRenderer();
	
	while(true)
	{
		//todo: get inputs
		
		display.displayFrame(window);
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
