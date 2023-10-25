from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.arrays import vbo
import pygame
import numpy as np
import time

# Vertex shader code
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader code
FRAGMENT_SHADER = """
#version 330 core
uniform float time;
uniform vec2 resolution;

out vec4 fragColor;

const int numCircles = 20;

void main()
{
    vec2 uv = gl_FragCoord.xy / resolution.xy;
    
    vec3 color = vec3(0.0);
    
    for (int i = 0; i < numCircles; i++)
    {
        // Calculate position and size of each circle
        float radius = 0.2 + 0.2 * sin(float(i) / numCircles * 3.14159 + time);
        vec2 center = vec2(0.5 + 0.3 * cos(float(i) + time), 0.5 + 0.2 * sin(float(i) - time));
        
        // Calculate distance from pixel to the circle
        float distance = length(uv - center) / radius;
        
        // Smoothly blend colors based on distance
        color += vec3(0.3, 0.7, 1.0) * (1.0 - smoothstep(0.93, 0.95, distance));
    }
    
    fragColor = vec4(color, 1.0);
}
"""

def main():
    pygame.init()
    
    # Set window size
    width, height = 800, 600
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    
    # Compile shaders and create program
    shader = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))
    glUseProgram(shader)
    
    # Set up vertex buffer object (VBO)
    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position)
    
    # Get uniform locations
    time_loc = glGetUniformLocation(shader, "time")
    resolution_loc = glGetUniformLocation(shader, "resolution")
    
    # Main loop
    start_time = time.time()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Set uniform values
        current_time = time.time() - start_time
        glUseProgram(shader)
        glUniform1f(time_loc, current_time)
        glUniform2f(resolution_loc, width, height)
        
        # Draw the VBO
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        # Update screen
        pygame.display.flip()

if __name__ == '__main__':
    main()
