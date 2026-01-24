import * as THREE from 'three';

/**
 * Calculates dynamic agent speed based on reward values.
 * Phase 10/13: Calibrated for ultra slow-mo enjoyability.
 */
export const calculateCinematicSpeed = (reward: number): number => {
    if (reward < 0) return 0.2; // Slow-mo near danger
    if (reward > 40) return 1.0; // Stabilized turbo
    return 0.15; // Ultra slow-mo for standard positive rewards
};

/**
 * Generates a camera offset relative to the agent position.
 * Phase 13/14: Precise horizontal centering and Midline stability.
 */
export const getCameraOffset = (progress: number): THREE.Vector3 => {
    // Dynamic X shift to keep Step 0 anchored left initially
    const xOffset = -5 + (progress * 0.2);

    // Controlled swerve for "roller coaster" feel
    const swerve = Math.sin(progress * 0.4) * 6;

    // Phase 14: Recalibrated vertical centering.
    // By setting vertical to 0 and looking at agentPos, 
    // the path resides in the horizontal midline.
    const vertical = 0;

    return new THREE.Vector3(xOffset, vertical, 15 + swerve);
};
