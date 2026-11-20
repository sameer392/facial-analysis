"use client";

import { useEffect, useState } from "react";

/** The API token is entered once, on the Settings page, and kept in this browser. */
export const TOKEN_KEY = "facial.apiToken";

export function readToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return ""; // storage blocked: the token is treated as not set
  }
}

export function writeToken(value: string): void {
  try {
    if (value === "") window.localStorage.removeItem(TOKEN_KEY);
    else window.localStorage.setItem(TOKEN_KEY, value);
  } catch {
    // storage blocked: nothing is saved
  }
}

/** The stored token, and whether it has been read yet (it is read after the first render). */
export function useApiToken(): [string, (value: string) => void, boolean] {
  const [token, setToken] = useState("");
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setToken(readToken());
    setLoaded(true);
  }, []);
  const save = (value: string) => {
    writeToken(value.trim());
    setToken(value.trim());
  };
  return [token, save, loaded];
}
