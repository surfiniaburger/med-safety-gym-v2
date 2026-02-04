import { useEffect, useState } from 'react';

export interface StreamSnapshot {
    timestamp: number;
    scores: Record<string, number>;
    step_index?: number;
}

export const useGauntletStream = (sessionId: string | null) => {
    const [streamData, setStreamData] = useState<{ rewards: number[], metrics: any[], snapshots: StreamSnapshot[] }>({ 
        rewards: [], 
        metrics: [],
        snapshots: []
    });
    const [isConnected, setIsConnected] = useState(false);

    useEffect(() => {
        if (!sessionId) return;

        // Determine WS URL (handle dev vs prod)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host; // e.g. localhost:3000 or localhost:8000
        // If running on port 3000 (Vite), we might need to proxy or point to 8000
        const wsUrl = `${protocol}//${host.replace('3000', '8000')}/ws/gauntlet/${sessionId}`;

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setIsConnected(true);
            console.log('Connected to Gauntlet Stream');
        };

        ws.onmessage = (event) => {
            try {
                const data: StreamSnapshot = JSON.parse(event.data);
                
                // Map snapshot to Gauntlet format
                // Assuming 'root' score is the total reward
                const reward = data.scores?.root || 0;
                
                // Map specific failures based on scores
                // These thresholds should match the server config
                const metric = {
                    hallucination: (data.scores?.['grounding'] || 0) <= -20.0,
                    format_error: (data.scores?.['format'] || 0) <= 0.0,
                    inconsistency: (data.scores?.['inconsistency'] || 0) <= -15.0,
                    refusal: (data.scores?.['refusal'] || 0) > 0.0, // Refusal usually gives a small positive reward in some configs, or we check if it applied
                };

                setStreamData(prev => ({
                    rewards: [...prev.rewards, reward],
                    metrics: [...prev.metrics, metric],
                    snapshots: [...prev.snapshots, data]
                }));
            } catch (e) {
                console.error("Error parsing stream data", e);
            }
        };

        ws.onclose = () => setIsConnected(false);

        return () => ws.close();
    }, [sessionId]);

    return { streamData, isConnected };
};
