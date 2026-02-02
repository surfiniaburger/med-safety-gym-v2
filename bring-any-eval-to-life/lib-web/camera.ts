import * as THREE from 'three';
import { PathGeometryType } from './path-generation';

export type CameraProfile = 'follow' | 'first-person' | 'birds-eye';

/**
 * Calculates dynamic agent speed based on reward values.
 */
export const calculateCinematicSpeed = (reward: number): number => {
    if (reward < 0) return 0.1; // Ultra slow-mo near danger for tension
    if (reward > 15) return 0.8; // Fast through high-confidence safe zones
    return 0.3; // Standard pace
};

/**
 * Generates a camera offset relative to the agent position.
 */
export const getCameraOffset = (
    progress: number, 
    type: PathGeometryType = 'linear',
    profile: CameraProfile = 'follow'
): THREE.Vector3 => {
    if (profile === 'birds-eye') {
        return type === 'spherical' 
            ? new THREE.Vector3(0, 80, 0) 
            : new THREE.Vector3(20, 60, 20);
    }

    if (profile === 'first-person') {
        return new THREE.Vector3(0, 0.5, 0); 
    }

    // Default: Follow Profile
    if (type === 'wormhole') {
        // Spiral offset: stay "behind" the agent in the spiral
        const angle = (progress / 10) * Math.PI * 4 - 0.5; // Slightly behind the current angle
        return new THREE.Vector3(
            Math.cos(angle) * 5,
            Math.sin(angle) * 5,
            -10
        );
    }
    
    if (type === 'spherical') {
        // Wide orbiting shot
        const orbitAngle = progress * 0.1;
        return new THREE.Vector3(
            Math.cos(orbitAngle) * 40,
            20,
            Math.sin(orbitAngle) * 40
        );
    }

    // Linear Path: Dynamic X shift to keep Step 0 anchored left initially
    const xOffset = -8 + (progress * 0.1);
    const swerve = Math.sin(progress * 0.4) * 4;
    const vertical = 2;

    return new THREE.Vector3(xOffset, vertical, 12 + swerve);
};
