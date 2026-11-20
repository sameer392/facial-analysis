"use client";

import { useEffect, useState } from "react";

export const TOKEN_KEY = "facial.apiToken";

/** The token is kept for this browser tab only. It is never put in the URL. */
export function useApiToken(): [string, (value: string) => void, boolean] {
  const [token, setToken] = useState("");
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    try {
      setToken(window.sessionStorage.getItem(TOKEN_KEY) ?? "");
    } catch {
      // storage unavailable (private window or blocked): the token must be typed each time
    }
    setLoaded(true);
  }, []);
  const save = (value: string) => {
    setToken(value);
    try {
      window.sessionStorage.setItem(TOKEN_KEY, value);
    } catch {
      // ignore: the token stays in memory for this page only
    }
  };
  return [token, save, loaded];
}

