"use client"
import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * AmbientShaderBackground
 * Adapted from XplainAI (MIT License - https://github.com/mahik504/XplainAI)
 * Customized for AstroVerse "Celestial Atelier" Light Aesthetic.
 * Fluid colors: Warm porcelain base, soft sky cyan, aerial blue, lavender, and warm champagne/coffee.
 */
export default function AmbientShaderBackground() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Check for prefers-reduced-motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return;
    }

    let renderer: THREE.WebGLRenderer | null = null;
    let animationFrameId: number;

    try {
      const scene = new THREE.Scene();
      const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'low-power'
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      const vertexShader = `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `;

      // Celestial Atelier fluid light shader
      const fragmentShader = `
        uniform float u_time;
        uniform vec2 u_resolution;
        varying vec2 vUv;

        // Simplex 2D noise
        vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }
        float snoise(vec2 v){
          const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                   -0.577350269189626, 0.024390243902439);
          vec2 i  = floor(v + dot(v, C.yy) );
          vec2 x0 = v -   i + dot(i, C.xx);
          vec2 i1;
          i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz;
          x12.xy -= i1;
          i = mod(i, 289.0);
          vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
            + i.x + vec3(0.0, i1.x, 1.0 ));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
            dot(x12.zw,x12.zw)), 0.0);
          m = m*m ;
          m = m*m ;
          vec3 x = 2.0 * fract(p * C.www) - 1.0;
          vec3 h = abs(x) - 0.5;
          vec3 ox = floor(x + 0.5);
          vec3 a0 = x - ox;
          m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
          vec3 g;
          g.x  = a0.x  * x0.x  + h.x  * x0.y;
          g.yz = a0.yz * x12.xz + h.yz * x12.yw;
          return 130.0 * dot(m, g);
        }

        void main() {
          vec2 uv = gl_FragCoord.xy / u_resolution.xy;
          float time = u_time * 0.12;

          // Celestial Atelier Color Palette (Normalized)
          vec3 c_porcelain = vec3(0.968, 0.952, 0.917); // #F7F3EA
          vec3 c_sky       = vec3(0.772, 0.929, 0.960); // #C5EDF5
          vec3 c_aerial    = vec3(0.513, 0.803, 0.886); // #83CDE2
          vec3 c_lavender  = vec3(0.862, 0.839, 0.968); // #DCD6F7
          vec3 c_champagne = vec3(0.910, 0.874, 0.847); // #E8DFD8

          // Fluid motion coordinates
          float n1 = snoise(uv * 1.8 + vec2(time * 0.2, time * 0.15));
          float n2 = snoise(uv * 2.5 - vec2(time * 0.18, time * 0.22) + n1 * 0.4);
          float n3 = snoise(uv * 3.2 + vec2(time * 0.12, -time * 0.15) + n2 * 0.3);

          // Blend into luminous flowing sky & warm paper
          vec3 color = mix(c_porcelain, c_sky, smoothstep(-0.6, 0.6, n1));
          color = mix(color, c_aerial, smoothstep(0.2, 0.9, n2) * 0.45);
          color = mix(color, c_lavender, smoothstep(-0.4, 0.5, n3) * 0.35);
          color = mix(color, c_champagne, smoothstep(0.4, 1.0, n1 * n2) * 0.25);

          // Subtle vignette
          float dist = distance(uv, vec2(0.5));
          color = mix(color, c_porcelain, smoothstep(0.4, 0.95, dist) * 0.35);

          gl_FragColor = vec4(color, 0.92);
        }
      `;

      const uniforms = {
        u_time: { value: 0.0 },
        u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
      };

      const material = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms,
        depthWrite: false,
        depthTest: false
      });

      const geometry = new THREE.PlaneGeometry(2, 2);
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      const clock = new THREE.Clock();

      const animate = () => {
        animationFrameId = requestAnimationFrame(animate);
        uniforms.u_time.value = clock.getElapsedTime();
        if (renderer) renderer.render(scene, camera);
      };

      animate();

      const handleResize = () => {
        if (!renderer) return;
        renderer.setSize(window.innerWidth, window.innerHeight);
        uniforms.u_resolution.value.set(window.innerWidth, window.innerHeight);
      };

      window.addEventListener('resize', handleResize);

      return () => {
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('resize', handleResize);
        if (renderer && renderer.domElement && container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
          renderer.dispose();
        }
        geometry.dispose();
        material.dispose();
      };
    } catch (err) {
      console.warn("Three.js shader initialization failed, using CSS fallback", err);
    }
  }, []);

  return (
    <div 
      ref={containerRef} 
      className="fixed inset-0 pointer-events-none z-0 bg-[#F7F3EA] overflow-hidden"
      style={{
        backgroundImage: `
          radial-gradient(at 10% 15%, rgba(197, 237, 245, 0.6) 0px, transparent 50%),
          radial-gradient(at 85% 10%, rgba(220, 214, 247, 0.45) 0px, transparent 50%),
          radial-gradient(at 50% 85%, rgba(131, 205, 226, 0.35) 0px, transparent 55%),
          radial-gradient(at 90% 75%, rgba(232, 223, 216, 0.5) 0px, transparent 50%)
        `
      }}
    />
  );
}
