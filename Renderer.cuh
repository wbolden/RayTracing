#ifndef RENDERER_H
#define RENDERER_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Renderer
{
public:
	Renderer(void);
	void setResolution(int width, int height);
	void setProjectionMode(bool orthographic);
	void renderFrame(int width, int height, cudaSurfaceObject_t pixels);

	~Renderer(void);

private:
	int* devPixels;
	bool orthographic;
	int rwidth, rheight;
};

#endif
