import * as THREE from 'three';
import { PathGeometryType } from './path-generation';

export type CameraProfile = 'follow' | 'first-person' | 'birds-eye';

/**
 * Calculates dynamic agent speed based on reward values.
 */
export const calculateCinematicSpeed = (reward: number): number => {
    if (reward < 0) return 0.2; // Slow-mo near danger
    if (reward > 40) return 1.0; // Stabilized turbo
    return 0.15; // Ultra slow-mo for standard positive rewards
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
        return new THREE.Vector3(20, 60, 20); // High overview
    }

    if (profile === 'first-person') {
        return new THREE.Vector3(0, 0.5, 0.1); // Just slightly above/ahead of agent
    }

    // Default: Follow Profile
    if (type === 'wormhole') {
        return new THREE.Vector3(0, 0, -15); // Look down the tunnel
    }
    
    if (type === 'spherical') {
        return new THREE.Vector3(30, 30, 30); // Wide shot of the sphere
    }

    // Dynamic X shift to keep Step 0 anchored left initially
    const xOffset = -5 + (progress * 0.2);
    const swerve = Math.sin(progress * 0.4) * 6;
    const vertical = 0;

    return new THREE.Vector3(xOffset, vertical, 15 + swerve);
};
