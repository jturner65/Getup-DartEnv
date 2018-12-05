#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # inheriting world class that facilitates modification to rendered data


from gym import error
import numpy as np
import os.path
import numpy as np

try:
    import pydart2 as pydart
    from pydart2 import pydart2_api as papi
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


#overriding world to add functionality to display info on screen
class myWorld(pydart.World):
    def __init__(self, dt, skelFullPath=None):
        pydart.World.__init__(self, dt, skelFullPath)
        self.leftOffset = 10
        self.lineOffset = 20

    #set a ref to the environment so the world can query for current display data
    def set2BotEnv(self, env):
        self.env = env

    #passes render interface - perhaps can be used to render skeletons with appropriate colors?
    def render_with_ri(self, ri):
        ri.push()
        ri.pushState()
        self.env.renderSkels(ri)
        ri.popState()
        ri.pop()

    #called from pydart2/gui/opengl/scene.py
    def draw_with_ri(self, ri):
        ri.push()
        ri.pushState()
        self.env.dispScrText(ri)
        ri.popState()
        ri.pop()

    #overriding world.render so i can control it
    def render(self,
               render_markers=True,
               render_contacts=True,
               render_contact_size=0.01,
               render_contact_force_scale=-0.005):

        #papi.world__render(self.id)        #renders via c++ code, see drawWorld() in pydart2_draw.cpp

        if render_markers:
            self.render_markers()

        if render_contacts:
            self.render_contacts(render_contact_size,
                                 render_contact_force_scale)







    #this is render in pydart2/gui/opengl/scene.py - looks for methods in "sim" (which is world) and calls if exists
    # def render(self, sim=None):
    #     GL.glEnable(GL.GL_DEPTH_TEST)
    #     GL.glClearColor(0.98, 0.98, 0.98, 0.0)
    #     GL.glClearColor(1.0, 1.0, 1.0, 1.0)
    #     GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    #     GL.glLoadIdentity()
    #     # glTranslate(0.0, -0.2, self.zoom)  # Camera
    #     GL.glTranslate(*self.tb.trans)
    #     GL.glMultMatrixf(self.tb.matrix)

    #     if sim is None:
    #         return

    #     if hasattr(sim, "render"):
    #         sim.render()

    #     self.renderer.enable("COLOR_MATERIAL")
    #     if hasattr(sim, "render_with_ri"):
    #         sim.render_with_ri(self.renderer)

    #     self.enable2D()
    #     if hasattr(sim, "draw_with_ri"):
    #         sim.draw_with_ri(self.renderer)
    #         self.renderer.draw_text([-100, -100], "")
    #     self.disable2D()

       