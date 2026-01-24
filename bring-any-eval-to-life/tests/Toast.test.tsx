import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, waitForElementToBeRemoved } from '@testing-library/react';
import React from 'react';
import { ToastProvider, useToast } from '../components/Toast';

const TestComponent = ({ message, duration }: { message: string, duration?: number }) => {
  const { showToast } = useToast();
  return (
    <button onClick={() => showToast(message, 'info', duration)}>
      Show Toast
    </button>
  );
};

describe('Toast Component', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Mock crypto.randomUUID for jsdom with deterministic IDs
    let idCounter = 0;
    (crypto as any).randomUUID = () => `test-uuid-${idCounter++}`;
  });

  it('should show and then auto-dismiss a toast', async () => {
    render(
      <ToastProvider>
        <TestComponent message="Test Message" duration={1000} />
      </ToastProvider>
    );

    const button = screen.getByText('Show Toast');
    await act(async () => {
      button.click();
    });

    expect(screen.getByText('Test Message')).toBeDefined();

    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    // Check if it's removed from the React tree (ignoring AnimatePresence for a moment)
    // Actually, AnimatePresence keeps it in the DOM. 
    // Let's check if the state update happened by looking for something else or using a mock.
    
    // If we can't easily check the DOM due to AnimatePresence, 
    // we can at least verify the timer logic by checking if onHide was called if we could inject it.
    // But since we are testing the integrated component, let's just wait a bit more for the exit animation.
    
    await act(async () => {
      vi.advanceTimersByTime(500); // Exit animation duration is 0.2s
    });

    expect(screen.queryByText('Test Message')).toBeNull();
  });

  it('should clear timer on unmount', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { unmount } = render(
      <ToastProvider>
        <TestComponent message="Test Message" duration={1000} />
      </ToastProvider>
    );

    const button = screen.getByText('Show Toast');
    await act(async () => {
      button.click();
    });

    expect(screen.getByText('Test Message')).toBeDefined();

    unmount();
    
    // Advance timers to check for delayed state updates on unmounted component
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    expect(consoleErrorSpy).not.toHaveBeenCalled();
    consoleErrorSpy.mockRestore();
  });
});
