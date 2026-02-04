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
    MeshDistortMaterial,
    MeshWobbleMaterial,
    Float,
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
    PlayIcon as PlayIconOutline,
    MagnifyingGlassPlusIcon,
    ArrowPathIcon,
    AdjustmentsHorizontalIcon,
    ChevronDownIcon
} from '@heroicons/react/24/outline';
import { calculateCinematicSpeed, getCameraOffset, CameraProfile } from '../../lib-web/camera';
import {
    WARMUP_START_X,
    WARMUP_TARGET_X,
    WARMUP_PURPLE_RADIUS,
    WARMUP_GREEN_RADIUS,
    WARMUP_GREEN_ORBIT_RADIUS,
    WARMUP_GREEN_ORBIT_SPEED,
    WARMUP_ENTRY_LERP_FACTOR
} from '../../lib-web/gauntlet-constants';
import { StepMetrics } from '../../lib-web/extraction';
import { generatePathPoints, PathGeometryType } from '../../lib-web/path-generation';

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
    const starCount = 4000;
    const [positions, colors, sizes] = useMemo(() => {
        const coords = new Float32Array(starCount * 3);
        const cols = new Float32Array(starCount * 3);
        const szs = new Float32Array(starCount);
        const range = 400;

        const colorOptions = [
            new THREE.Color("#ffffff"),
            new THREE.Color("#4dabf7"), // Blue
            new THREE.Color("#845ef7"), // Purple
            new THREE.Color("#fab005"), // Gold
        ];

        for (let i = 0; i < starCount; i++) {
            coords[i * 3] = (Math.random() - 0.5) * range;
            coords[i * 3 + 1] = (Math.random() - 0.5) * range;
            coords[i * 3 + 2] = (Math.random() - 0.5) * range;

            const color = colorOptions[Math.floor(Math.random() * (Math.random() > 0.9 ? colorOptions.length : 1))];
            cols[i * 3] = color.r;
            cols[i * 3 + 1] = color.g;
            cols[i * 3 + 2] = color.b;

            szs[i] = Math.random() * 0.15 + 0.05;
        }
        return [coords, cols, szs];
    }, []);

    const starsRef = useRef<THREE.Points>(null);
    useFrame((state) => {
        if (starsRef.current) {
            starsRef.current.rotation.y += 0.0002;
            starsRef.current.rotation.x += 0.0001;
            // Subtle parallax based on camera position
            starsRef.current.position.x = state.camera.position.x * 0.05;
            starsRef.current.position.y = state.camera.position.y * 0.05;
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
                <bufferAttribute
                    attach="attributes-color"
                    count={starCount}
                    array={colors}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.15}
                vertexColors
                transparent
                opacity={0.6}
                sizeAttenuation
                blending={THREE.AdditiveBlending}
            />
        </points>
    );
};

interface WarmupAgentsProps {
    state: GauntletState;
    isFailedNode: boolean;
}

const WarmupAgents = ({ state, isFailedNode }: WarmupAgentsProps) => {
    const purpleRef = useRef<THREE.Group>(null);
    const greenRef = useRef<THREE.Group>(null);
    const pulseRef = useRef<THREE.PointLight>(null);

    useFrame((threeState) => {
        const time = threeState.clock.getElapsedTime();

        if (state === GauntletState.WARMUP_ENTRY) {
            // Purple model agent enters from far left
            if (purpleRef.current) {
                purpleRef.current.position.x = THREE.MathUtils.lerp(purpleRef.current.position.x, WARMUP_TARGET_X, WARMUP_ENTRY_LERP_FACTOR);
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
            {/* Purple Model Agent - Visible during warmup OR when frozen due to failure */}
            {(state === GauntletState.WARMUP_ENTRY || state === GauntletState.EVALUATION_PULSE || state === GauntletState.TRANSITION || (state === GauntletState.TRAJECTORY_ACTIVE && isFailedNode)) && (
                <group ref={purpleRef} position={[WARMUP_TARGET_X, 0, 0]}>
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

interface BarrierNodeProps {
    position: THREE.Vector3;
    active: boolean;
    intensity: number;
}

const BarrierNode = ({ position, active, intensity }: BarrierNodeProps) => {
    const lightRef = useRef<THREE.PointLight>(null);

    useFrame((state) => {
        if (active && lightRef.current) {
            const time = state.clock.getElapsedTime();
            lightRef.current.intensity = (Math.sin(time * 10) * 5 + 15) * intensity;
        }
    });

    return (
        <group position={position}>
            {active && (
                <Float speed={8 * intensity} rotationIntensity={0.5 * intensity} floatIntensity={0.5 * intensity}>
                    <mesh rotation={[0, 0, 0]}>
                        <planeGeometry args={[5, 5]} />
                        <MeshWobbleMaterial
                            color="#fa5252"
                            speed={4 * intensity}
                            factor={0.8 * intensity}
                            transparent
                            opacity={0.3}
                            side={THREE.DoubleSide}
                        />
                    </mesh>
                    <pointLight ref={lightRef} distance={8} intensity={15 * intensity} color="#fa5252" />
                </Float>
            )}
        </group>
    );
};

interface GauntletViewProps {
    rewards: number[];
    metrics?: StepMetrics[];
    activeStepIndex: number;
    solvedNodes: number[];
    onIntervene: (index: number) => void;
    onActiveStepChange: (index: number) => void;
    onClose: () => void;
    onComplete?: () => void;
    initialPathType?: PathGeometryType;
    accentColor?: string;
    snapshots?: any[]; // Added for Neural Diagnostics
}

interface PathAgentProps {
    points: THREE.Vector3[];
    rewards: number[];
    currentIndex: number;
    isPaused: boolean;
    simSpeed: number;
    onProgress: (index: number) => void;
}

const PathAgent = ({
    points,
    rewards,
    currentIndex,
    isPaused,
    simSpeed,
    onProgress
}: PathAgentProps) => {
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
            const baseSpeed = calculateCinematicSpeed(currentReward);
            const nextProgress = progress + delta * baseSpeed * simSpeed;
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

interface CinematicCameraProps {
    enabled: boolean;
    type: PathGeometryType;
    profile: CameraProfile;
    points: THREE.Vector3[];
}

const CinematicCamera = ({ enabled, type, profile, points }: CinematicCameraProps) => {
    const { agentPosition, agentProgress } = useGauntlet();

    useFrame((state) => {
        if (enabled && agentPosition && points.length > 0) {
            const offset = getCameraOffset(agentProgress, type, profile);
            const targetPos = agentPosition.clone().add(offset);

            // Smoothly lerp camera to target position
            state.camera.position.lerp(targetPos, 0.05);

            if (profile === 'first-person') {
                // Look ahead along the path
                const nextIndex = Math.min(Math.floor(agentProgress) + 1, points.length - 1);
                const lookTarget = points[nextIndex];
                state.camera.lookAt(lookTarget);
            } else {
                state.camera.lookAt(agentPosition);
            }
        }
    });

    return null;
};

interface NeuralPathwayProps {
    points: THREE.Vector3[];
    rewards: number[];
    solvedNodes: number[];
    pathType: PathGeometryType;
    intensity: number;
    color: string;
    activeStepIndex: number;
}

const NeuralPathway = ({ points, rewards, solvedNodes, pathType, intensity, color, activeStepIndex }: NeuralPathwayProps) => {
    const { agentProgress } = useGauntlet();
    const curve = useMemo(() => new THREE.CatmullRomCurve3(points), [points]);

    return (
        <group>
            {/* Decorative Tunnel for Wormhole */}
            {pathType === 'wormhole' && (
                <mesh>
                    <tubeGeometry args={[curve, 100, 4, 12, false]} />
                    <MeshDistortMaterial
                        color={color}
                        speed={3 * intensity}
                        distort={0.2 * intensity}
                        transparent
                        opacity={0.08}
                        wireframe
                    />
                </mesh>
            )}

            {/* Decorative Globe for Spherical */}
            {pathType === 'spherical' && (
                <group>
                    <Float speed={2 * intensity} rotationIntensity={0.2 * intensity} floatIntensity={0.2 * intensity}>
                        <Sphere args={[25, 48, 48]}>
                            <meshStandardMaterial
                                color={color}
                                transparent
                                opacity={0.05}
                                wireframe
                            />
                        </Sphere>
                    </Float>
                    {/* Inner Core */}
                    <Sphere args={[5, 32, 32]}>
                        <MeshDistortMaterial
                            color={color}
                            speed={5 * intensity}
                            distort={0.5 * intensity}
                            transparent
                            opacity={0.2}
                        />
                    </Sphere>
                    <pointLight intensity={10 * intensity} color={color} distance={20} />
                </group>
            )}

            {/* Living Path: Base line */}
            <Line
                points={points}
                color={pathType === 'linear' ? "#1a1b1e" : "#343a40"}
                lineWidth={1}
                transparent
                opacity={0.5}
            />
            {/* Active glowing segment */}
            <Line
                points={points.slice(0, Math.ceil(agentProgress + 1))}
                color={color}
                lineWidth={3}
                transparent
                opacity={0.9}
            />
            {points.map((p, i) => {
                const reward = rewards[i];
                const isSolved = solvedNodes.includes(i);
                const isFailed = reward < 0 && !isSolved;
                const isActive = i === activeStepIndex;
                const shouldShowLabel = isActive || i % 5 === 0 || isFailed;

                return (
                    <group key={i} position={p}>
                        <Sphere args={[0.25, 16, 16]}>
                            <meshStandardMaterial
                                color={isFailed ? "#fa5252" : isSolved ? "#40c057" : color}
                                emissive={isFailed ? "#fa5252" : isSolved ? "#40c057" : color}
                                emissiveIntensity={isFailed ? 3 : 0.8}
                            />
                        </Sphere>
                        {isFailed && (
                            <Float speed={5} rotationIntensity={2} floatIntensity={2}>
                                <Text
                                    position={[0, 1.5, 0]}
                                    fontSize={0.5}
                                    color="#fa5252"
                                >
                                    CRITICAL_ERROR
                                </Text>
                            </Float>
                        )}
                        {shouldShowLabel && (
                            <Text
                                position={[0, -0.8, 0]}
                                fontSize={isActive ? 0.3 : 0.2}
                                color={isActive ? "#ffffff" : "#5c5f66"}
                            >
                                NODE_{i}
                            </Text>
                        )}
                    </group>
                );
            })}
        </group>
    );
};

const NeuralDiagnostics = ({ snapshot, isOpen, onClose }: { snapshot: any, isOpen: boolean, onClose: () => void }) => {
    if (!isOpen || !snapshot) return null;

    return (
        <div className="absolute right-0 top-20 bottom-24 md:bottom-20 w-[85%] md:w-80 bg-black/80 backdrop-blur-xl border-l border-white/10 p-4 md:p-6 overflow-y-auto z-40 rounded-l-3xl shadow-2xl">
            <div className="flex items-center justify-between mb-4 md:mb-6">
                <div className="flex items-center gap-2">
                    <MagnifyingGlassPlusIcon className="w-4 h-4 md:w-5 md:h-5 text-blue-400" />
                    <h3 className="text-sm md:text-lg font-bold text-white uppercase tracking-widest">Neural Diagnostics</h3>
                </div>
                <button
                    onClick={onClose}
                    data-testid="close-diagnostics-btn"
                    className="p-1 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-white transition-colors"
                >
                    <XMarkIcon className="w-5 h-5" />
                </button>
            </div>

            <div className="flex flex-col gap-1">
                {Object.entries(snapshot.scores || {}).map(([key, value]) => {
                    const val = Number(value);
                    const isNegative = val < 0;
                    return (
                        <div key={key} className="group flex justify-between items-center p-2 md:p-3 rounded-xl hover:bg-white/5 transition-colors border border-transparent hover:border-white/5">
                            <div className="flex flex-col">
                                <span className="text-zinc-400 font-mono text-[8px] md:text-[10px] uppercase tracking-wider group-hover:text-white transition-colors">{key.replace('.', ' > ')}</span>
                            </div>
                            <span className={`font-mono text-[10px] md:text-xs font-bold px-1.5 md:px-2 py-0.5 md:py-1 rounded-md ${isNegative
                                ? 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
                                : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                {val.toFixed(1)}
                            </span>
                        </div>
                    );
                })}
            </div>

            <div className="mt-4 md:mt-6 pt-4 md:pt-6 border-t border-white/10">
                <div className="text-[8px] md:text-[10px] text-zinc-600 font-mono uppercase text-center">
                    Snapshot Timestamp: {new Date((snapshot.timestamp || 0) * 1000).toLocaleTimeString()}
                </div>
            </div>
        </div>
    );
};

export const GauntletView: React.FC<GauntletViewProps> = ({
    rewards,
    metrics,
    activeStepIndex,
    solvedNodes,
    onIntervene,
    onActiveStepChange,
    onClose,
    onComplete,
    initialPathType = 'linear',
    accentColor = '#4dabf7',
    snapshots
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

    const [gameState, setGameState] = useState<GauntletState>(GauntletState.WARMUP_ENTRY);
    const [cameraMode, setCameraMode] = useState<'cinematic' | 'manual'>('cinematic');
    const [cameraProfile, setCameraProfile] = useState<CameraProfile>('follow');
    const [pathType, setPathType] = useState<PathGeometryType>(initialPathType);
    const [neuralIntensity, setNeuralIntensity] = useState(0.5);
    const [simSpeed, setSimSpeed] = useState(1.0);
    const [showSettings, setShowSettings] = useState(false);
    const [showDiagnostics, setShowDiagnostics] = useState(false);

    // Keyboard Navigation Support
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            const navKeys = ['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', '+', '-'];
            if (navKeys.includes(e.key) && gameState === GauntletState.TRAJECTORY_ACTIVE) {
                setCameraMode('manual');
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [gameState]);

    const points = useMemo(() => {
        return generatePathPoints(rewards, pathType, PATH_SPACING, REWARD_Y_SCALE);
    }, [rewards, pathType]);

    const currentReward = rewards[activeStepIndex];
    const currentMetrics = metrics?.[activeStepIndex];
    const isFailedNode = currentReward < 0 && !solvedNodes.includes(activeStepIndex);
    const [isInternalPaused, setIsInternalPaused] = useState(isFailedNode);
    const [isManualPaused, setIsManualPaused] = useState(false);
    const [showRestored, setShowRestored] = useState(false);
    const showFailureUI = isFailedNode && gameState === GauntletState.TRAJECTORY_ACTIVE;

    const failureTitle = useMemo(() => {
        if (!currentMetrics) return "Safety Violation Detected";
        if (currentMetrics.hallucination) return "Hallucination Detected";
        if (currentMetrics.format_error) return "Format Error Detected";
        if (currentMetrics.inconsistency) return "Inconsistency Detected";
        if (currentMetrics.refusal) return "Refusal Detected";
        return "Safety Violation Detected";
    }, [currentMetrics]);

    const failureDescription = useMemo(() => {
        if (!currentMetrics) return "The neural pathway has encountered a critical alignment error. Manual intervention is required to realign weights and restore safety protocols.";
        if (currentMetrics.hallucination) return "The model has generated medically inaccurate or fabricated information. This poses a critical risk to patient safety.";
        if (currentMetrics.format_error) return "The model failed to adhere to the required output format, potentially disrupting downstream clinical systems.";
        if (currentMetrics.inconsistency) return "The model's reasoning is inconsistent with its final recommendation, indicating a failure in logical processing.";
        if (currentMetrics.refusal) return "The model has refused to provide a necessary medical recommendation, which may delay critical care.";
        return "The neural pathway has encountered a critical alignment error. Manual intervention is required to realign weights and restore safety protocols.";
    }, [currentMetrics]);

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
            {/* Neural Diagnostics Panel */}
            <NeuralDiagnostics
                snapshot={snapshots?.[activeStepIndex]}
                isOpen={showDiagnostics && (gameState === GauntletState.TRAJECTORY_ACTIVE || gameState === GauntletState.TRANSITION)}
                onClose={() => setShowDiagnostics(false)}
            />

            {/* Cinematic Camera Control */}
            <Canvas shadows dpr={[1, 2]}>
                <GauntletContext.Provider value={contextValue}>
                    <CinematicCamera enabled={cameraMode === 'cinematic'} type={pathType} profile={cameraProfile} points={points} />
                    <PerspectiveCamera makeDefault position={[-20, 10, 20]} />
                    <Starfield />

                    <WarmupAgents state={gameState} isFailedNode={isFailedNode} />

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
                            <NeuralPathway
                                points={points}
                                rewards={rewards}
                                solvedNodes={solvedNodes}
                                pathType={pathType}
                                intensity={neuralIntensity}
                                color={accentColor}
                                activeStepIndex={activeStepIndex}
                            />

                            {points.map((p, i) => (
                                rewards[i] < 0 && (
                                    <BarrierNode key={`barrier-${i}`} position={p} active={!solvedNodes.includes(i)} intensity={neuralIntensity} />
                                )
                            ))}

                            <PathAgent
                                points={points}
                                rewards={rewards}
                                currentIndex={activeStepIndex}
                                isPaused={isInternalPaused}
                                simSpeed={simSpeed}
                                onProgress={onActiveStepChange}
                            />

                            <ContactShadows opacity={0.4} scale={100} blur={2} far={10} resolution={256} color="#000000" />
                        </>
                    )}
                </GauntletContext.Provider>
            </Canvas>

            {/* UI Overlays */}
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-2 pointer-events-none">
                <div className="text-white/20 text-[8px] font-mono uppercase">
                    Gauntlet Mount OK ({rewards.length} nodes)
                </div>

                {gameState === GauntletState.TRAJECTORY_ACTIVE && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-black/60 backdrop-blur-md border border-white/10 rounded-full px-4 py-2 flex items-center gap-3 shadow-2xl"
                    >
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        <span className="text-[10px] font-mono text-blue-400 font-bold uppercase tracking-widest">
                            Architect: {
                                pathType === 'wormhole' ? "Visualizing Data Turbulence via Wormhole" :
                                    pathType === 'spherical' ? "Mapping Global Safety via Fibonacci Sphere" :
                                        "Standard Linear Trajectory Active"
                            }
                        </span>
                    </motion.div>
                )}
            </div>

            <div className="absolute top-0 left-0 w-full p-4 md:p-6 flex justify-between items-start pointer-events-none z-50">
                <div className="flex flex-col gap-1 pointer-events-auto">
                    <div className="flex items-center gap-2 px-2 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[8px] md:text-[10px] font-bold uppercase tracking-widest w-fit">
                        <BoltIcon className="w-3 h-3" /> Gauntlet Engine V1.0
                    </div>
                    <h1 className="text-lg md:text-2xl font-black text-white">Neural Path</h1>
                </div>

                <div className="flex items-center gap-2 pointer-events-auto">
                    {/* Settings Dropdown */}
                    <div className="relative">
                        <button
                            onClick={() => setShowSettings(!showSettings)}
                            className={`p-2 rounded-xl border transition-all flex items-center gap-2 ${showSettings
                                ? 'bg-blue-500 border-blue-400 text-white'
                                : 'bg-white/5 border-white/10 text-white/40 hover:text-white hover:bg-white/10'
                                }`}
                        >
                            <AdjustmentsHorizontalIcon className="w-5 h-5 md:w-6 md:h-6" />
                            <span className="hidden md:block text-[10px] font-bold uppercase tracking-widest">Controls</span>
                        </button>

                        <AnimatePresence>
                            {showSettings && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                                    className="absolute right-0 mt-2 w-64 md:w-72 bg-zinc-900/90 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-2xl flex flex-col gap-4"
                                >
                                    {/* Path Type */}
                                    <div className="flex flex-col gap-2">
                                        <label className="text-[8px] text-zinc-500 uppercase font-bold tracking-widest">Trajectory Type</label>
                                        <div className="grid grid-cols-3 gap-1">
                                            {(['linear', 'wormhole', 'spherical'] as PathGeometryType[]).map((type) => (
                                                <button
                                                    key={type}
                                                    onClick={() => setPathType(type)}
                                                    className={`px-2 py-1.5 rounded-lg text-[9px] font-bold uppercase tracking-tighter transition-all border ${pathType === type
                                                        ? 'bg-blue-500 border-blue-400 text-white'
                                                        : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10'
                                                        }`}
                                                >
                                                    {type}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Camera Profile */}
                                    <div className="flex flex-col gap-2">
                                        <label className="text-[8px] text-zinc-500 uppercase font-bold tracking-widest">Camera View</label>
                                        <div className="grid grid-cols-3 gap-1">
                                            {(['follow', 'first-person', 'birds-eye'] as CameraProfile[]).map((profile) => (
                                                <button
                                                    key={profile}
                                                    onClick={() => {
                                                        setCameraProfile(profile);
                                                        setCameraMode('cinematic');
                                                    }}
                                                    className={`px-2 py-1.5 rounded-lg text-[9px] font-bold uppercase tracking-tighter transition-all border ${cameraProfile === profile
                                                        ? 'bg-emerald-500 border-emerald-400 text-white'
                                                        : 'bg-white/5 border-white/10 text-white/40 hover:bg-white/10'
                                                        }`}
                                                >
                                                    {profile.replace('-', ' ')}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <div className="h-px bg-white/10" />

                                    {/* Sliders */}
                                    <div className="flex flex-col gap-3">
                                        <div className="flex flex-col gap-1">
                                            <div className="flex justify-between items-center">
                                                <label htmlFor="neural-intensity-slider" className="text-[8px] text-zinc-500 uppercase font-bold tracking-widest">Neural Intensity</label>
                                                <span className="text-[8px] font-mono text-blue-400 font-bold">{(neuralIntensity * 100).toFixed(0)}%</span>
                                            </div>
                                            <input
                                                id="neural-intensity-slider"
                                                type="range" min="0" max="1" step="0.01"
                                                value={neuralIntensity}
                                                onChange={(e) => setNeuralIntensity(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>

                                        <div className="flex flex-col gap-1">
                                            <div className="flex justify-between items-center">
                                                <label htmlFor="sim-speed-slider" className="text-[8px] text-zinc-500 uppercase font-bold tracking-widest">Sim Speed</label>
                                                <span className="text-[8px] font-mono text-blue-400 font-bold">{simSpeed.toFixed(1)}x</span>
                                            </div>
                                            <input
                                                id="sim-speed-slider"
                                                type="range" min="0.1" max="3.0" step="0.1"
                                                value={simSpeed}
                                                onChange={(e) => setSimSpeed(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                                            />
                                        </div>
                                    </div>

                                    <div className="h-px bg-white/10" />

                                    {/* Diagnostics Toggle */}
                                    <button
                                        onClick={() => setShowDiagnostics(!showDiagnostics)}
                                        className={`flex items-center justify-center gap-2 px-4 py-2 rounded-xl border transition-all text-[10px] font-bold uppercase tracking-widest ${showDiagnostics
                                            ? 'bg-blue-500 border-blue-400 text-white'
                                            : 'bg-white/5 border-white/10 text-white/60 hover:text-white'
                                            }`}
                                    >
                                        <MagnifyingGlassPlusIcon className="w-4 h-4" />
                                        {showDiagnostics ? 'Hide Diagnostics' : 'Show Diagnostics'}
                                    </button>

                                    {/* Camera Mode Toggle */}
                                    <button
                                        onClick={() => setCameraMode(cameraMode === 'cinematic' ? 'manual' : 'cinematic')}
                                        className={`flex items-center justify-center gap-2 px-4 py-2 rounded-xl border transition-all text-[10px] font-bold uppercase tracking-widest ${cameraMode === 'manual'
                                            ? 'bg-amber-500/10 border-amber-500/20 text-amber-500'
                                            : 'bg-white/5 border-white/10 text-white/60 hover:text-white'
                                            }`}
                                    >
                                        {cameraMode === 'manual' ? <ArrowPathIcon className="w-4 h-4" /> : <MagnifyingGlassPlusIcon className="w-4 h-4" />}
                                        {cameraMode === 'manual' ? 'Manual Mode' : 'Free Camera'}
                                    </button>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    <button
                        onClick={onClose}
                        className="p-2 rounded-xl bg-white/5 border border-white/10 text-white/40 hover:text-white hover:bg-white/10 transition-all"
                    >
                        <XMarkIcon className="w-5 h-5 md:w-6 md:h-6" />
                    </button>
                </div>
            </div>

            {/* Glitch Overlay */}
            {showFailureUI && <GlitchOverlay />}

            {/* Intervention Required Modal */}
            <AnimatePresence>
                {showFailureUI && (
                    <motion.div
                        key="intervention-modal"
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
                                <h2 className="text-3xl font-black text-white mb-3">{failureTitle}</h2>
                                <p className="text-zinc-400 text-sm mb-10 leading-relaxed font-medium">
                                    {failureDescription}
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

            {/* Alignment Restored Overlay */}
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
            <div className="absolute bottom-6 md:bottom-12 left-1/2 -translate-x-1/2 w-full max-w-md md:max-w-xl px-4 pointer-events-none">
                <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-3 md:p-4 flex items-center justify-between pointer-events-auto shadow-2xl">
                    <div className="flex items-center gap-3 md:gap-4">
                        <div className="flex flex-col">
                            <span className="hidden sm:block text-[8px] md:text-[10px] text-zinc-500 uppercase font-bold tracking-widest leading-none mb-1">Position</span>
                            <span className="text-white font-mono text-sm md:text-xl">STEP {activeStepIndex}</span>
                        </div>
                        <div className="h-6 md:h-8 w-px bg-white/10" />
                        <div className="flex flex-col">
                            <span className="hidden sm:block text-[8px] md:text-[10px] text-zinc-500 uppercase font-bold tracking-widest leading-none mb-1">Status</span>
                            <span className={`text-[10px] md:text-sm font-bold flex items-center gap-1.5 ${isInternalPaused ? 'text-rose-500' : 'text-sky-400'}`}>
                                <div className={`w-1.5 h-1.5 md:w-2 md:h-2 rounded-full ${isInternalPaused ? 'bg-rose-500 animate-pulse' : 'bg-sky-400 animate-ping'}`} />
                                <span className="hidden xs:inline">{isInternalPaused ? 'INTERVENTION' : 'ACTIVE'}</span>
                            </span>
                        </div>
                    </div>

                    <div className="flex items-center gap-2 md:gap-3">
                        <button
                            onClick={() => setIsManualPaused(!isManualPaused)}
                            className="p-1.5 md:p-2 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white transition-all"
                            title="Pause Simulation"
                        >
                            {isManualPaused ? <PlayIconOutline className="w-4 h-4 md:w-5 md:h-5" /> : <PauseIcon className="w-4 h-4 md:w-5 md:h-5" />}
                        </button>

                        {rewards[activeStepIndex] >= 0 && (
                            <button
                                onClick={() => onIntervene(activeStepIndex)}
                                className="flex items-center gap-1.5 px-2 md:px-4 py-1.5 md:py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[8px] md:text-[10px] font-bold uppercase tracking-widest hover:bg-emerald-500/20 transition-all"
                            >
                                <ShieldCheckIcon className="w-3 md:w-3.5 h-3 md:h-3.5" />
                                <span className="hidden sm:inline">Sanity Test</span>
                            </button>
                        )}

                        <div className="flex items-center gap-1 md:gap-2 overflow-x-auto max-w-[80px] md:max-w-[150px] scrollbar-hide">
                            {rewards.map((_, i) => (
                                <div
                                    key={i}
                                    className={`h-1 flex-shrink-0 rounded-full transition-all duration-300 ${i === activeStepIndex ? 'w-4 md:w-8 bg-white' :
                                        solvedNodes.includes(i) ? 'w-1 md:w-2 bg-emerald-500' :
                                            rewards[i] < 0 ? 'w-1 md:w-2 bg-rose-500' : 'w-1 md:w-2 bg-white/10'
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
