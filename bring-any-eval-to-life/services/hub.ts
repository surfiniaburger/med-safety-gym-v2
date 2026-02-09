import { EvaluationArtifact } from './github';

const HUB_URL = import.meta.env.VITE_RENDER_HUB || "https://med-safety-hub.onrender.com";

interface HubSessionMetadata {
    safety_score?: number;
    status?: 'SAFE' | 'UNSAFE' | 'UNKNOWN';
    summary?: string;
    run_type?: string;
    task_id?: string;
    [key: string]: any;
}

interface HubSession {
    session_id: string;
    metadata: HubSessionMetadata;
    step_count: number;
}

export async function fetchHubArtifacts(): Promise<EvaluationArtifact[]> {
    try {
        const response = await fetch(`${HUB_URL}/gauntlet/sessions`);
        if (!response.ok) {
            console.warn('Hub session fetch failed:', response.statusText);
            return [];
        }

        const data = await response.json();
        const sessions: HubSession[] = data.sessions || [];

        return sessions.map(session => ({
            id: session.session_id,
            name: `Session: ${session.session_id}`,
            path: `db/${session.session_id}`,
            sha: session.session_id,
            url: `${HUB_URL}/gauntlet/data/${session.session_id}`,
            html_url: '',
            download_url: '',
            content: {
                safety_score: session.metadata?.safety_score ?? 0.5,
                status: (session.metadata?.status || 'UNKNOWN') as 'SAFE' | 'UNSAFE' | 'UNKNOWN',
                summary: session.metadata?.summary || `Database Session with ${session.step_count} steps.`,
                run_type: session.metadata?.run_type,
                task_id: session.metadata?.task_id,
                metadata: session.metadata
            }
        }));
    } catch (error) {
        console.error('Failed to fetch hub artifacts:', error);
        return [];
    }
}

/**
 * Fetches snapshot data for a specific Hub session.
 * Phase 17: Data Format Unification
 */
export async function fetchHubSessionData(sessionId: string): Promise<any[]> {
    try {
        const response = await fetch(`${HUB_URL}/gauntlet/data/${sessionId}`);
        if (!response.ok) {
            console.warn(`Hub session data fetch failed for ${sessionId}:`, response.statusText);
            return [];
        }
        const data = await response.json();
        return data.snapshots || data.results || [];
    } catch (error) {
        console.error(`Failed to fetch hub session data for ${sessionId}:`, error);
        return [];
    }
}
