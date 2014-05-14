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
	void createLens(int width, int height);
	void createSphericalLens(int width, int height, int radius);
	void setProjectionMode(bool orthographic);
	void renderFrame(cudaSurfaceObject_t pixels);

	~Renderer(void);

private:
	bool orthographic;
	int renderWidth, renderHeight;
	dim3 blockSize;
	dim3 gridSize;
};

struct Sphere
{
	float3 position;
	float radius;
	float reflectivity;
};

struct CameraInfo
{
	float3 position;
	float3 viewpos;
	float3 rotation;
};

#endif
