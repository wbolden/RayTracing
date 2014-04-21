#include <Windows.h>
#include "Timer.h"


Timer::Timer(void)
{
	startTime = getMS();
	elapsedTime = getMS() - startTime;
	frameCount = 0;
}

void Timer::start(void)
{
	startTime = getMS();
}

void Timer::stop(void)
{
	elapsedTime = getMS() - startTime;
	frameCount++;
}

long long Timer::getElapsed(void)
{
	return elapsedTime;
}

float Timer::getElapsedMS(void)
{
	return (float)elapsedTime/1000.0f;
}

long long Timer::getFrameCount(void)
{
	return frameCount;
}

float Timer::getFPS(void)
{
	return 1/getElapsedMS();
}

long long Timer::getMS(void)
{
	static LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);

	return (1000LL * now.QuadPart)/ frequency.QuadPart;
}


Timer::~Timer(void)
{
}
