import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import App from '../App';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
        {children}
    </QueryClientProvider>
);

describe('SFT vs GRPO Data Fetching Integration', () => {
    beforeEach(() => {
        vi.resetAllMocks();
        localStorage.clear();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('fetches both GitHub Reports and Hub Live Sessions and identifies them', async () => {
        const mockGitHubFiles = [
            {
                name: 'github_report.json',
                sha: 'sha-gh',
                type: 'file',
                path: 'results/github_report.json',
                url: 'https://api.github.com/repos/owner/repo/contents/results/github_report.json'
            }
        ];

        const mockHubSessions = {
            sessions: [
                {
                    session_id: 'hub_session_123',
                    metadata: { run_type: 'sft', task_id: 'task_1', safety_score: 0.95 },
                    step_count: 50
                }
            ]
        };

        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch
            .mockImplementation((url: string) => {
                if (url.includes('api.github.com')) {
                    return Promise.resolve({
                        ok: true,
                        headers: new Headers({ 'ETag': 'etag-1' }),
                        json: async () => mockGitHubFiles,
                    });
                }
                if (url.includes('/gauntlet/sessions')) {
                    return Promise.resolve({
                        ok: true,
                        json: async () => mockHubSessions,
                    });
                }
                if (url.includes('github_report.json')) {
                    return Promise.resolve({
                        ok: true,
                        json: async () => ({
                            content: btoa(JSON.stringify({ safety_score: 0.8, status: 'SAFE' }))
                        }),
                    });
                }
                return Promise.reject(new Error('Unknown URL'));
            });

        render(<App />, { wrapper });

        // Wait for both cards to appear
        await waitFor(() => {
            expect(screen.getByTestId('artifact-card-sha-gh')).toBeDefined();
            expect(screen.getByTestId('artifact-card-hub_session_123')).toBeDefined();
        });

        // Verify Source Badges
        expect(screen.getByText(/GitHub Report/i)).toBeDefined();
        expect(screen.getByText(/Live Session/i)).toBeDefined();
    });
});
