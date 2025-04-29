export interface DartScore {
  id: number;
  created_at: string;
  player: "robot" | "human";
  score: number;
  round: number;
}

export interface GameState {
  robotScore: number;
  humanScore: number;
  currentRound: number;
  lastThrow?: {
    player: "robot" | "human";
    score: number;
  };
}
