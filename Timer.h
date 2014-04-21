#ifndef TIMER_H
#define TIMER_H
class Timer
{
public:
	Timer(void);

	void start(void);
	void stop(void);

	long long getElapsed(void);
	float getElapsedMS(void);
	long long getFrameCount(void);
	float getFPS(void);

	~Timer(void);

private:
	long long elapsedTime;
	long long startTime;
	long long frameCount;

	long long getMS(void);
};

#endif