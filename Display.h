#ifndef DISPLAY_H
#define DISPLAY_H

#include "Renderer.cuh"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <gl\GL.h>
#include <gl\GLU.h>

class Display
{
public:
	Display(GLFWwindow* window);
	
	void initWindow();
	void resizeWindow(GLFWwindow* window, int width, int height);
	void createRenderer();
	void displayFrame(GLFWwindow* window);
	
	~Display(void);

private:
	int width, height;
	Renderer render;
	GLuint vtex;
	cudaGraphicsResource_t cudavResource;
};

#endif
