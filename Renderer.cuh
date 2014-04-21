#ifndef RENDERER_H
#define RENDERER_H

class Renderer
{
public:
	Renderer(void);
	void setResolution(int width, int height);
	void setProjectionMode(bool orthographic);
	void renderFrame(int width, int height, int* pixels);	//returns the bitmap representing a frame

	~Renderer(void);

private:
	int* devPixels;
	bool orthographic;
};

#endif
