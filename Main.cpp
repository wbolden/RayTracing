#include <windows.h>
#include <windowsx.h>
#include "Renderer.cuh"
#include "Timer.h"

#define WIDTH 512
#define HEIGHT 512

HWND hWnd;
HDC hdc;
WNDCLASSEX wc;

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
int run(HINSTANCE &hinstance);

int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hiprev, LPSTR cmd, int cmdShow)
{
	ZeroMemory(&wc, sizeof(WNDCLASSEX));

    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hinstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "WindowClass";

    RegisterClassEx(&wc);

	RECT wr = {0, 0, WIDTH, HEIGHT};
	
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    hWnd = CreateWindowEx(NULL,
                          "WindowClass",
                          "Ray Tracing",
                          WS_OVERLAPPEDWINDOW,
                          300,
                          300,
                          wr.right - wr.left,
                          wr.bottom - wr.top,
                          NULL,
                          NULL,
						  hinstance,
                          NULL);

	hdc = GetDC(hWnd);

    ShowWindow(hWnd, cmdShow);

	int result = run(hinstance);

	ReleaseDC(hWnd, hdc);

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


LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
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

    return DefWindowProc (hWnd, message, wParam, lParam);
}