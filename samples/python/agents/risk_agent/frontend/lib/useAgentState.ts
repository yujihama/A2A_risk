"use client";

import useSWR from "swr";
import { useEffect, useState, useRef } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const WS_URL = BACKEND_URL.replace(/^http/, "ws") + "/ws";

async function fetcher(url: string) {
  console.log("[Fetcher] Fetching:", url);
  const res = await fetch(url);
  if (!res.ok) {
    console.error("[Fetcher] Failed to fetch:", res.status, res.statusText);
    throw new Error(`HTTP error! status: ${res.status}`);
  }
  console.log("[Fetcher] Fetched successfully.");
  return res.json();
}

export function useAgentState() {
  const isMountedRef = useRef(true);

  const { data: init, error: swrError } = useSWR(`${BACKEND_URL}/state/initial`, fetcher);
  const [state, setState] = useState<any | null>(null);

  useEffect(() => {
    if (swrError) {
      console.error("[useAgentState] SWR Error fetching initial state:", swrError);
    }
  }, [swrError]);

  useEffect(() => {
    if (init) {
      console.log("[useAgentState] Setting initial state from SWR:", Object.keys(init));
      setState(init);
    }
  }, [init]);

  useEffect(() => {
    isMountedRef.current = true;
    console.log("[useAgentState] useEffect for WebSocket mounting...");

    console.log(`[useAgentState] Attempting to connect WebSocket: ${WS_URL}`);
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("[WebSocket] Connection opened.");
    };

    ws.onmessage = (event) => {
      try {
        const receivedState = JSON.parse(event.data);
        console.log("[WebSocket] Message received, setting state. Keys:", Object.keys(receivedState));
        if (isMountedRef.current) {
          setState(receivedState);
        }
      } catch (e) {
        console.error("[WebSocket] Error parsing message:", e);
      }
    };

    ws.onerror = (event) => {
      console.error("[WebSocket] Error:", event);
    };

    ws.onclose = (event) => {
      console.log(`[WebSocket] Connection closed. Code: ${event.code}, Reason: ${event.reason}, Clean: ${event.wasClean}`);
    };

    return () => {
      console.log("[useAgentState] useEffect for WebSocket unmounting... Closing WebSocket.");
      isMountedRef.current = false;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
         ws.close();
         console.log("[WebSocket] close() called.");
      }
    };
  }, []);

  return state;
} 