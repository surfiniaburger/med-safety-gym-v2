import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import {
    OrbitControls,
    PerspectiveCamera,
    Text,
    Sphere,
    Trail,
    Line,
    ContactShadows,
} from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import {
    XMarkIcon,
    ExclamationTriangleIcon,
    BoltIcon,
    ArrowRightIcon,
    ShieldCheckIcon,
    PauseIcon,
    PlayIcon as PlayIconOutline
} from '@heroicons/react/24/outline';
import { calculateCinematicSpeed, getCameraOffset } from '../../lib-web/camera';
import { 
    WARMUP_START_X, 
    WARMUP_TARGET_X, 
    WARMUP_PURPLE_RADIUS, 
    WARMUP_GREEN_RADIUS, 
    WARMUP_GREEN_ORBIT_RADIUS,
    WARMUP_GREEN_ORBIT_SPEED 
} from '../../lib-web/gauntlet-constants';

enum GauntletState {
    WARMUP_ENTRY,
    EVALUATION_PULSE,
    TRANSITION,
    TRAJECTORY_ACTIVE
}

// Phase 14: Constants extracted from magic numbers
const PATH_SPACING = 3;
const REWARD_Y_SCALE = 0.4;

interface GauntletContextValue {
    agentPosition: THREE.Vector3;
    agentProgress: number;
    setAgentState: (pos: THREE.Vector3, progress: number) => void;
    onComplete?: () => void;
}

const GauntletContext = React.createContext<GauntletContextValue | null>(null);

const useGauntlet = () => {
    const context = React.useContext(GauntletContext);
    if (!context) throw new Error("useGauntlet must be used within GauntletProvider");
    return context;
};

const Starfield = () => {
    const starCount = 3000;
    const positions = useMemo(() => {
        const coords = new Float32Array(starCount * 3);
        const range = 300; // Increased range for centering flexibility
        for (let i = 0; i < starCount; i++) {
            coords[i * 3] = (Math.random() - 0.5) * range;
            coords[i * 3 + 1] = (Math.random() - 0.5) * range;
            coords[i * 3 + 2] = (Math.random() - 0.5) * range;
        }
        return coords;
    }, []);

    const starsRef = useRef<THREE.Points>(null);
    useFrame((state) => {
        if (starsRef.current) {
            starsRef.current.rotation.y += 0.0001;
            starsRef.current.rotation.x += 0.00005;
        }
    });

    return (
        <points ref={starsRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={starCount}
                    array={positions}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial size={0.12} color="#ffffff" transparent opacity={0.4} sizeAttenuation />
        </points>
    );
};

const WarmupAgents = ({ state }: { state: GauntletState }) => {
    const purpleRef = useRef<THREE.Group>(null);
    const greenRef = useRef<THREE.Group>(null);
    const pulseRef = useRef<THREE.PointLight>(null);

    useFrame((threeState) => {
        const time = threeState.clock.getElapsedTime();

        if (state === GauntletState.WARMUP_ENTRY) {
            // Purple model agent enters from far left
            if (purpleRef.current) {
                purpleRef.current.position.x = THREE.MathUtils.lerp(purpleRef.current.position.x, WARMUP_TARGET_X, 0.02);
                purpleRef.current.position.y = Math.sin(time * 2) * 0.5;
            }
        }

        if (state === GauntletState.EVALUATION_PULSE || state === GauntletState.TRANSITION) {
            // Green evaluator agent circles the purple one
            if (greenRef.current && purpleRef.current) {
                const angle = time * WARMUP_GREEN_ORBIT_SPEED;
                greenRef.current.position.x = purpleRef.current.position.x + Math.cos(angle) * WARMUP_GREEN_ORBIT_RADIUS;
                greenRef.current.position.z = purpleRef.current.position.z + Math.sin(angle) * WARMUP_GREEN_ORBIT_RADIUS;
                greenRef.current.position.y = Math.sin(time * 4) * 1;
            }
            // Pulsing evaluation light
            if (pulseRef.current) {
                pulseRef.current.intensity = Math.sin(time * 10) * 5 + 5;
            }
        }
    });

    return (
        <group>
            {/* Purple Model Agent - Only visible during early warmup */}
            {(state === GauntletState.WARMUP_ENTRY || state === GauntletState.EVALUATION_PULSE) && (
                <group ref={purpleRef} position={[WARMUP_START_X, 0, 0]}>
                    <Sphere args={[WARMUP_PURPLE_RADIUS, 32, 32]}>
                        <meshStandardMaterial color="#845ef7" emissive="#845ef7" emissiveIntensity={2} />
                    </Sphere>
                    <pointLight intensity={2} color="#845ef7" distance={10} />
                </group>
            )}
            {/* Green Evaluator Agent */}
            {(state === GauntletState.EVALUATION_PULSE || state === GauntletState.TRANSITION) && (
                <group ref={greenRef}>
                    <Sphere args={[WARMUP_GREEN_RADIUS, 16, 16]}>
                        <meshStandardMaterial color="#40c057" emissive="#40c057" emissiveIntensity={3} />
                    </Sphere>
                    <pointLight ref={pulseRef} distance={8} color="#40c057" />
                    <Trail width={1} length={4} color={new THREE.Color("#40c057")}>
                        <mesh />
                    </Trail>
                </group>
            )}
        </group>
    );
};

const GlitchOverlay = () => (
    <div className="absolute inset-x-0 inset-y-0 z-[60] pointer-events-none glitch-overlay">
        <div className="absolute inset-0 bg-rose-500/10 mix-blend-color-dodge animate-pulse" />
        <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />
        <div className="absolute inset-0 flex flex-col justify-between opacity-30">
            <div className="h-[1px] w-full bg-rose-500 animate-[scan_2s_linear_infinite]" />
            <div className="h-[1px] w-full bg-rose-500 animate-[scan_3s_linear_infinite_reverse]" />
        </div>
        <style dangerouslySetInnerHTML={{
            __html: `
            @keyframes scan {
                from { transform: translateY(-100vh); }
                to { transform: translateY(100vh); }
            }
        `}} />
    </div>
);

const BarrierNode = ({ position, active }: { position: THREE.Vector3, active: boolean }) => (
    <group position={position}>
        <mesh rotation={[0, 0, 0]}>
            <planeGeometry args={[4, 4]} />
            <meshStandardMaterial
                color="#fa5252"
                transparent
                opacity={active ? 0.3 : 0}
                side={THREE.DoubleSide}
                wireframe={active}
            />
        </mesh>
        {active && (
            <pointLight distance={5} intensity={5} color="#fa5252" />
        )}
    </group>
);

interface GauntletViewProps {
    rewards: number[];
    activeStepIndex: number;
    solvedNodes: number[];
    onIntervene: (index: number) => void;
    onActiveStepChange: (index: number) => void;
    onClose: () => void;
    onComplete?: () => void;
}

const PathAgent = ({
    points,
    rewards,
    currentIndex,
    isPaused,
    onProgress
}: {
    points: THREE.Vector3[],
    rewards: number[],
    currentIndex: number,
    isPaused: boolean,
    onProgress: (index: number) => void
}) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const [progress, setProgress] = useState(currentIndex);

    useEffect(() => {
        if (Math.abs(progress - currentIndex) > 0.5) {
            setProgress(currentIndex);
        }
    }, [currentIndex]);

    const { setAgentState, onComplete } = useGauntlet();

    useFrame((state, delta) => {
        if (isPaused) return;

        if (progress < points.length - 1) {
            const currentReward = rewards[Math.floor(progress)] || 0;
            const speedMultiplier = calculateCinematicSpeed(currentReward);
            const nextProgress = progress + delta * speedMultiplier;
            setProgress(nextProgress);

            const floorIndex = Math.floor(nextProgress);
            if (floorIndex > currentIndex) {
                onProgress(floorIndex);
            }
            if (nextProgress >= points.length - 1 && typeof onComplete === 'function') {
                onComplete();
            }
        }
    });

    const currentPos = useMemo(() => {
        if (!points || points.length === 0) return new THREE.Vector3(0, 0, 0);
        const idx = Math.floor(progress);
        const nextIdx = Math.min(idx + 1, points.length - 1);
        const lerp = progress % 1;

        if (!points[idx] || !points[nextIdx]) return new THREE.Vector3(0, 0, 0);
        return new THREE.Vector3().lerpVectors(points[idx], points[nextIdx], lerp);
    }, [progress, points]);

    // Update context for camera tracking
    useEffect(() => {
        setAgentState(currentPos, progress);
    }, [currentPos, progress, setAgentState]);

    return (
        <group position={currentPos}>
            <Sphere args={[0.5, 32, 32]} ref={meshRef}>
                <meshStandardMaterial
                    color="#4dabf7"
                    emissive="#4dabf7"
                    emissiveIntensity={2}
                    roughness={0}
                    metalness={1}
                />
            </Sphere>
            <pointLight intensity={2} color="#4dabf7" distance={10} />
            <Trail
                width={2}
                length={5}
                color={new THREE.Color("#4dabf7")}
                attenuation={(t) => t * t}
            >
                <mesh />
            </Trail>
        </group>
    );
};

const CinematicCamera = () => {
    const { agentPosition, agentProgress } = useGauntlet();

    useFrame((state) => {
        if (agentPosition) {
            const offset = getCameraOffset(agentProgress);
            const targetPos = agentPosition.clone().add(offset);

            // Smoothly lerp camera to target position
            state.camera.position.lerp(targetPos, 0.05);
            state.camera.lookAt(agentPosition);
        }
    });

    return null;
};

const NeuralPathway = ({ points, rewards, solvedNodes }: { points: THREE.Vector3[], rewards: number[], solvedNodes: number[] }) => {
    const { agentProgress } = useGauntlet();

    return (
        <group>
            {/* Living Path: Base line */}
            <Line
                points={points}
                color="#1a1b1e"
                lineWidth={1}
            />
            {/* Active glowing segment */}
            <Line
                points={points.slice(0, Math.ceil(agentProgress + 1))}
                color="#4dabf7"
                lineWidth={2}
                transparent
                opacity={0.8}
            />
            {points.map((p, i) => {
                const reward = rewards[i];
                const isSolved = solvedNodes.includes(i);
                const isFailed = reward < 0 && !isSolved;

                return (
                    <group key={i} position={p}>
                        <Sphere args={[0.2, 16, 16]}>
                            <meshStandardMaterial
                                color={isFailed ? "#fa5252" : isSolved ? "#40c057" : "#339af0"}
                                emissive={isFailed ? "#fa5252" : isSolved ? "#40c057" : "#339af0"}
                                emissiveIntensity={isFailed ? 2 : 0.5}
                            />
                        </Sphere>
                        {isFailed && (
                            <Text
                                position={[0, 1.2, 0]}
                                fontSize={0.4}
                                color="#fa5252"
                                font="monospace"
                            >
                                DANGER
                            </Text>
                        )}
                        <Text
                            position={[0, -0.6, 0]}
                            fontSize={0.2}
                            color="#5c5f66"
                        >
                            Step {i}
                        </Text>
                    </group>
                );
            })}
        </group>
    );
};

export const GauntletView: React.FC<GauntletViewProps> = ({
    rewards,
    activeStepIndex,
    solvedNodes,
    onIntervene,
    onActiveStepChange,
    onClose,
    onComplete
}) => {
    if (!rewards || rewards.length === 0) {
        return (
            <div className="w-full h-full bg-[#050505] flex flex-col items-center justify-center p-6 text-center">
                <ExclamationTriangleIcon className="w-16 h-16 text-zinc-800 mb-6" />
                <h2 className="text-2xl font-black text-white mb-2">Insufficient Neural Data</h2>
                <p className="text-zinc-500 max-w-sm mb-8 leading-relaxed">
                    This evaluation artifact does not contain the required reward metrics for 3D trajectory mapping.
                </p>
                <div className="flex flex-col gap-3 w-full max-w-xs">
                    <button
                        onClick={() => onIntervene(0)}
                        className="w-full flex items-center justify-center gap-2 bg-white text-black py-3 rounded-xl font-bold hover:bg-zinc-200 transition-colors"
                    >
                        <BoltIcon className="w-4 h-4" />
                        Skip to Simulator
                    </button>
                    <button
                        onClick={onClose}
                        className="w-full py-3 rounded-xl border border-white/10 text-white/60 font-medium hover:bg-white/5 transition-colors"
                    >
                        Exit to Selection
                    </button>
                </div>
            </div>
        );
    }

    const points = useMemo(() => {
        // Phase 14: Midline centering - rewards are scaled to +/- range but centered on Y=0
        return rewards.map((r, i) => new THREE.Vector3(i * PATH_SPACING, r * REWARD_Y_SCALE, 0));
    }, [rewards]);

    const currentReward = rewards[activeStepIndex];
    const isFailedNode = currentReward < 0 && !solvedNodes.includes(activeStepIndex);
    const [isInternalPaused, setIsInternalPaused] = useState(isFailedNode);
    const [isManualPaused, setIsManualPaused] = useState(false);
    const [showRestored, setShowRestored] = useState(false);
    const [gameState, setGameState] = useState<GauntletState>(GauntletState.WARMUP_ENTRY);

    // Phase 11: Warmup Sequence Timing
    useEffect(() => {
        const entryTimeout = setTimeout(() => setGameState(GauntletState.EVALUATION_PULSE), 2000);
        const pulseTimeout = setTimeout(() => setGameState(GauntletState.TRANSITION), 5000);
        const trajectoryTimeout = setTimeout(() => setGameState(GauntletState.TRAJECTORY_ACTIVE), 6500);

        return () => {
            clearTimeout(entryTimeout);
            clearTimeout(pulseTimeout);
            clearTimeout(trajectoryTimeout);
        };
    }, []);

    useEffect(() => {
        setIsInternalPaused(isFailedNode || gameState !== GauntletState.TRAJECTORY_ACTIVE || isManualPaused);
    }, [isFailedNode, activeStepIndex, gameState, isManualPaused]);

    // Phase 4: Detect resolution
    const lastSolvedCount = useRef(solvedNodes.length);
    useEffect(() => {
        if (solvedNodes.length > lastSolvedCount.current) {
            setShowRestored(true);
            setTimeout(() => setShowRestored(false), 3000);
        }
        lastSolvedCount.current = solvedNodes.length;
    }, [solvedNodes]);

    // State for context
    const [agentPosition, setAgentPosition] = useState(new THREE.Vector3());
    const [agentProgress, setAgentProgress] = useState(0);

    // Memoize setAgentState to prevent re-creating on every render
    const setAgentState = React.useCallback((pos: THREE.Vector3, progress: number) => {
        setAgentPosition(pos);
        setAgentProgress(progress);
    }, []);

    // Memoize context value to prevent unnecessary re-renders of consumers
    const contextValue = React.useMemo(() => ({
        agentPosition,
        agentProgress,
        setAgentState,
        onComplete
    }), [agentPosition, agentProgress, setAgentState, onComplete]);

    return (
        <div className="w-full h-full relative font-sans select-none overflow-hidden">
            {/* Cinematic Camera Control */}
            <Canvas shadows dpr={[1, 2]}>
                <GauntletContext.Provider value={contextValue}>
                    <CinematicCamera />
                    <PerspectiveCamera makeDefault position={[-20, 10, 20]} />
                    <Starfield />

                    <WarmupAgents state={gameState} />

                    <OrbitControls
                        enablePan={true}
                        minDistance={5}
                        maxDistance={100}
                        makeDefault
                        dampingFactor={0.05}
                        enableDamping
                    />

                    <color attach="background" args={['#050505']} />
                    <fog attach="fog" args={['#050505', 10, 100]} />

                    <ambientLight intensity={0.5} />
                    <pointLight position={[10, 10, 10]} intensity={1.5} color="#4dabf7" castShadow />
                    <directionalLight position={[-5, 5, 5]} intensity={0.5} color="#ffffff" />

                    {(gameState === GauntletState.TRAJECTORY_ACTIVE || gameState === GauntletState.TRANSITION) && (
                        <>
                            <NeuralPathway points={points} rewards={rewards} solvedNodes={solvedNodes} />

                            {points.map((p, i) => (
                                rewards[i] < 0 && (
                                    <BarrierNode key={`barrier-${i}`} position={p} active={!solvedNodes.includes(i)} />
                                )
                            ))}

                            <PathAgent
                                points={points}
                                rewards={rewards}
                                currentIndex={activeStepIndex}
                                isPaused={isInternalPaused}
                                onProgress={onActiveStepChange}
                            />

                            <ContactShadows opacity={0.4} scale={100} blur={2} far={10} resolution={256} color="#000000" />
                        </>
                    )}
                </GauntletContext.Provider>
            </Canvas>

            {/* UI Overlays */}
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 text-white/20 text-[8px] font-mono pointer-events-none uppercase">
                Gauntlet Mount OK ({rewards.length} nodes)
            </div>

            <div className="absolute top-0 left-0 w-full p-6 flex justify-between items-start pointer-events-none">
                <div className="flex flex-col gap-1 pointer-events-auto">
                    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-bold uppercase tracking-widest">
                        <BoltIcon className="w-3 h-3" /> Gauntlet Engine V1.0
                    </div>
                    <h1 className="text-2xl font-black text-white">Neural Path Visualization</h1>
                </div>

                <button
                    onClick={onClose}
                    className="p-2 rounded-full bg-white/5 border border-white/10 text-white/40 hover:text-white hover:bg-white/10 transition-all pointer-events-auto"
                >
                    <XMarkIcon className="w-6 h-6" />
                </button>
            </div>

            {/* Glitch Overlay */}
            {isFailedNode && <GlitchOverlay />}

            {/* Intervention Required Modal */}
            <AnimatePresence>
                {isFailedNode && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        className="absolute inset-0 flex items-center justify-center p-6 pointer-events-none z-[70]"
                    >
                        <div className="max-w-md w-full bg-zinc-950/60 backdrop-blur-2xl border border-rose-500/30 rounded-[2.5rem] p-10 shadow-[0_0_80px_rgba(244,63,94,0.15)] pointer-events-auto relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-b from-rose-500/5 to-transparent pointer-events-none" />

                            <div className="flex flex-col items-center text-center relative z-10">
                                <div className="w-20 h-20 rounded-3xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-8 rotate-12">
                                    <ExclamationTriangleIcon className="w-10 h-10 text-rose-500 -rotate-12" />
                                </div>

                                <span className="text-rose-500 font-mono text-[10px] font-bold uppercase tracking-[0.3em] mb-4">Autopilot Failure: Node {activeStepIndex}</span>
                                <h2 className="text-3xl font-black text-white mb-3">Hallucination Detected</h2>
                                <p className="text-zinc-400 text-sm mb-10 leading-relaxed font-medium">
                                    The neural pathway has encountered a critical alignment error. Manual intervention is required to realign weights and restore safety protocols.
                                </p>

                                <button
                                    onClick={() => onIntervene(activeStepIndex)}
                                    className="w-full flex items-center justify-center gap-3 bg-white text-black hover:bg-zinc-200 py-5 rounded-2xl font-black transition-all group scale-100 hover:scale-[1.02] active:scale-[0.98]"
                                >
                                    <BoltIcon className="w-5 h-5" />
                                    Launch Neuro-Sim v4.2
                                    <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                                </button>

                                <div className="mt-6 flex items-center gap-4 text-[9px] text-zinc-600 font-bold uppercase tracking-widest">
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                                        Barrier Active
                                    </div>
                                    <div className="w-1 h-1 rounded-full bg-zinc-800" />
                                    <span>Payload: En-Route</span>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Phase 4: Alignment Restored Overlay */}
            <AnimatePresence>
                {showRestored && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.2 }}
                        className="absolute inset-0 flex items-center justify-center z-[100] pointer-events-none"
                    >
                        <div className="flex flex-col items-center gap-6">
                            <div className="w-24 h-24 rounded-full bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center">
                                <ShieldCheckIcon className="w-12 h-12 text-emerald-400" />
                            </div>
                            <div className="text-center">
                                <h2 className="text-4xl font-black text-white tracking-widest uppercase">Alignment Restored</h2>
                                <p className="text-emerald-400 font-mono text-xs mt-2 font-bold tracking-[0.2em] animate-pulse">RESUMING MISSION PROTOCOLS</p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Status Bar */}
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 w-full max-w-xl pointer-events-none">
                <div className="bg-black/40 backdrop-blur-md border border-white/5 rounded-2xl p-4 flex items-center justify-between pointer-events-auto">
                    <div className="flex items-center gap-4">
                        <div className="flex flex-col">
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest leading-none mb-1">Position</span>
                            <span className="text-white font-mono text-xl">STEP {activeStepIndex}</span>
                        </div>
                        <div className="h-8 w-px bg-white/10" />
                        <div className="flex flex-col">
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest leading-none mb-1">Status</span>
                            <span className={`text-sm font-bold flex items-center gap-1.5 ${isInternalPaused ? 'text-rose-500' : 'text-sky-400'}`}>
                                <div className={`w-2 h-2 rounded-full ${isInternalPaused ? 'bg-rose-500 animate-pulse' : 'bg-sky-400 animate-ping'}`} />
                                {isInternalPaused ? 'INTERVENTION REQ' : 'ACTIVE SIM'}
                            </span>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setIsManualPaused(!isManualPaused)}
                            className="p-2 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white transition-all pointer-events-auto"
                            title={isManualPaused ? "Resume Simulation" : "Pause Simulation"}
                        >
                            {isManualPaused ? <PlayIconOutline className="w-5 h-5" /> : <PauseIcon className="w-5 h-5" />}
                        </button>

                        {rewards[activeStepIndex] >= 0 && (
                            <button
                                onClick={() => onIntervene(activeStepIndex)}
                                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-widest hover:bg-emerald-500/20 transition-all pointer-events-auto"
                            >
                                <ShieldCheckIcon className="w-3.5 h-3.5" /> Sanity Test
                            </button>
                        )}
                        <div className="flex items-center gap-2 overflow-x-auto max-w-[150px] scrollbar-hide">
                            {rewards.map((_, i) => (
                                <div
                                    key={i}
                                    className={`h-1 flex-shrink-0 rounded-full transition-all duration-300 ${i === activeStepIndex ? 'w-8 bg-white' :
                                        solvedNodes.includes(i) ? 'w-2 bg-emerald-500' :
                                            rewards[i] < 0 ? 'w-2 bg-rose-500' : 'w-2 bg-white/10'
                                        }`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
