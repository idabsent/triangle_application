CFLAGS = -std=c++17
LDFLAGS = -lpthread -ldl -lvulkan -lglfw -lX11 -lXi -lXxf86vm

TriangleApplication: main.cpp
	glslangValidator -V100 shaders/shader.vert -o shaders/vert.spv && \
	glslangValidator -V100 shaders/shader.frag -o shaders/frag.spv && \
	g++-12 $(CFLAGS) -o TriangleApplication main.cpp $(LDFLAGS)

.PHONY: test clean

test: TriangleApplication
	make && ./TriangleApplication "Test application"

clean:
	rm -f TriangleApplication