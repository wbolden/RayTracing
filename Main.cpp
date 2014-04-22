#include <windows.h>
#include <windowsx.h>
#include "Renderer.cuh"
#include "Timer.h"

#define WIDTH 1280
#define HEIGHT 720

HWND hwnd;
HDC hdc;
WNDCLASSEX wc;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
int run(HINSTANCE &hinstance);

int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hiprev, LPSTR cmd, int cmdShow)
{
	ZeroMemory(&wc, sizeof(WNDCLASSEX));

    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hinstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "RayTracing";

    RegisterClassEx(&wc);

	RECT windowRect = {0, 0, WIDTH, HEIGHT};
	
    AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);

    hwnd = CreateWindowEx(NULL,
                          wc.lpszClassName,
                          "Ray Tracing",
                          WS_OVERLAPPEDWINDOW,
                          300,
                          300,
                          windowRect.right - windowRect.left,
                          windowRect.bottom - windowRect.top,
                          NULL,
                          NULL,
						  hinstance,
                          NULL);

	hdc = GetDC(hwnd);

    ShowWindow(hwnd, cmdShow);

	int result = run(hinstance);

	ReleaseDC(hwnd, hdc);

	return result;
}


int run(HINSTANCE &hinstance)
{
	Renderer render;
	Timer timer;
	MSG msg;
	int* pixels;

	render = Renderer();
	timer = Timer();
	render.setResolution(WIDTH, HEIGHT);
	pixels = new int[WIDTH*HEIGHT];

	while(true)
    {
        if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if(msg.message == WM_QUIT)
                break;
        }

 		timer.start();

		render.renderFrame(WIDTH, HEIGHT, pixels);

		HBITMAP frame = CreateBitmap(WIDTH, HEIGHT, 1, sizeof(int)*8, pixels);
		if(frame != NULL)
		{
			DrawState(hdc, NULL, NULL, (LPARAM)frame, NULL, 0, 0, 0, 0, DST_BITMAP);
		}
		DeleteObject(frame);

		timer.stop();

		float f = timer.getFPS();
		float fc = timer.getFrameCount();
    }

	 return msg.wParam;
}


LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch(message)
    {
        case WM_DESTROY:
            {
                PostQuitMessage(0);
                return 0;
				break;
            }
    }

    return DefWindowProc (hwnd, message, wParam, lParam);
}