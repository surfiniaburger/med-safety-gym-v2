import * as THREE from 'three';

export type PathGeometryType = 'linear' | 'wormhole' | 'spherical';

export const generatePathPoints = (
    rewards: number[], 
    type: PathGeometryType, 
    spacing: number = 3, 
    yScale: number = 0.4
): THREE.Vector3[] => {
    return rewards.map((r, i) => {
        if (type === 'wormhole') {
            // Cylindrical wrap: Z is progress, X/Y are the circle
            const angle = (i / rewards.length) * Math.PI * 4; // 2 full rotations
            const radius = 8 + r * yScale; // Base radius + reward distortion
            return new THREE.Vector3(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                i * spacing
            );
        } else if (type === 'spherical') {
            // Fibonacci Sphere mapping
            const phi = Math.acos(-1 + (2 * i) / rewards.length);
            const theta = Math.sqrt(rewards.length * Math.PI) * phi;
            const radius = 25 + r * yScale;
            return new THREE.Vector3(
                radius * Math.cos(theta) * Math.sin(phi),
                radius * Math.sin(theta) * Math.sin(phi),
                radius * Math.cos(phi)
            );
        }
        // Default: Linear
        return new THREE.Vector3(i * spacing, r * yScale, 0);
    });
};
