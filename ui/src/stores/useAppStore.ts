import { create } from 'zustand'

interface AppState {
  title: string
  setTitle: (title: string) => void
  reset: () => void
}

const initialState = {
  title: 'Hello World',
}

export const useAppStore = create<AppState>((set) => ({
  ...initialState,
  setTitle: (title) => set({ title }),
  reset: () => set(initialState),
}))
