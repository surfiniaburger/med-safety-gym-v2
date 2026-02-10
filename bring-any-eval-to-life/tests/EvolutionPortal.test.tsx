import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EvolutionPortal } from '../components/EvolutionPortal';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const createQueryClient = () => new QueryClient({
    defaultOptions: {
        queries: {
            retry: false,
        },
    },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={createQueryClient()}>
        {children}
    </QueryClientProvider>
);

describe('EvolutionPortal Component', () => {
    beforeEach(() => {
        vi.resetAllMocks();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('displays error UI when fetch fails', async () => {
        const globalFetch = vi.fn();
        global.fetch = globalFetch;

        globalFetch.mockResolvedValueOnce({
            ok: false,
            status: 500,
            statusText: 'Internal Server Error',
        });

        render(<EvolutionPortal taskId="test-task" onClose={() => {}} />, { wrapper });

        // Expect the error UI to be visible
        await waitFor(() => {
            expect(screen.getByText(/Evolution Data Error/i)).toBeDefined();
            expect(screen.getByText(/HTTP 500/i)).toBeDefined();
        }, { timeout: 5000 });
    });
});
