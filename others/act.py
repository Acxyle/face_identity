#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:12:54 2024

@author: acxyle-workstation
"""

import numpy as np
import matplotlib.pyplot as plt

# --- ReLU
x = np.concatenate([np.linspace(0,0,500), np.linspace(0,10,500)])

fig,ax = plt.subplots(figsize=(3,3))
ax.vlines(500, 0, 10, linestyle='--', colors='gray', alpha=0.5)
ax.hlines(0, 0, 1000, linestyle='--', colors='gray', alpha=0.5)

ax.plot(x, color='orange', linewidth=5)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig('ReLU.svg')

fig,ax = plt.subplots(figsize=(3,3))

ax.hlines(0, 0, 500, colors='purple', linewidth=5)
ax.hlines(10, 500, 1000, colors='purple', linewidth=5)

ax.vlines(500, 0, 10, linestyle='--', colors='gray', alpha=0.5)
ax.hlines(0, 0, 1000, linestyle='--', colors='gray', alpha=0.5)

ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig('ReLU_g.svg')

fig, ax = plt.subplots(figsize=(3,3))

ax.hlines(0, -1, 0, linewidth=5)
ax.hlines(1, 0, 1, linewidth=5)
ax.vlines(0, 0, 1, linewidth=5)

ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig('H.svg')

x = np.linspace(-3, 3, 1000)
y = 1/(1+(np.pi*x)**2)

fig, ax = plt.subplots(figsize=(3,3))
ax.plot(x, y, color='red', linestyle='--', linewidth=5)
ax.vlines(0, 0, 1, linestyle='--', colors='gray', alpha=0.5)
ax.hlines(0, -3, 3, linestyle='--', colors='gray', alpha=0.5)

x = np.linspace(-3, 3, 1000)
y = 1/np.pi*np.arctan(np.pi*x)+0.5

ax.plot(x, y, color='blue', linewidth=5)
ax.vlines(0, 0, 1, linestyle='--', colors='gray', alpha=0.5)
ax.hlines(0, -3, 3, linestyle='--', colors='gray', alpha=0.5)

ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig('Surrogate_g.svg')