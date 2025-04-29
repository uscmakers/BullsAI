"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { supabase } from "@/lib/supabase";
import type { DartScore, GameState } from "@/types/score";
import { BotIcon as Robot, Target, User } from "lucide-react";
import { useEffect, useState } from "react";
import { DatabaseSetupGuide } from "./database-setup-guide";

export function Scoreboard() {
  const [gameState, setGameState] = useState<GameState>({
    robotScore: 0,
    humanScore: 0,
    currentRound: 1,
  });
  const [recentScores, setRecentScores] = useState<DartScore[]>([]);
  const [loading, setLoading] = useState(true);

  // Add a new state for database setup status
  const [databaseReady, setDatabaseReady] = useState(true);

  useEffect(() => {
    // Initial fetch of scores
    const fetchScores = async () => {
      try {
        const { data, error } = await supabase
          .from("dart_scores")
          .select("*")
          .order("created_at", { ascending: false })
          .limit(10);

        if (error) {
          if (
            error.message.includes("relation") &&
            error.message.includes("does not exist")
          ) {
            setDatabaseReady(false);
          }
          throw error;
        }

        if (data) {
          setRecentScores(data as DartScore[]);

          // Calculate total scores
          const robotTotal = data
            .filter((score: DartScore) => score.player === "robot")
            .reduce((sum: number, score: DartScore) => sum + score.score, 0);

          const humanTotal = data
            .filter((score: DartScore) => score.player === "human")
            .reduce((sum: number, score: DartScore) => sum + score.score, 0);

          // Find the current round
          const maxRound =
            data.length > 0
              ? Math.max(...data.map((score: DartScore) => score.round))
              : 1;

          // Get the last throw
          const lastThrow =
            data.length > 0
              ? { player: data[0].player, score: data[0].score }
              : undefined;

          setGameState({
            robotScore: robotTotal,
            humanScore: humanTotal,
            currentRound: maxRound,
            lastThrow,
          });
        }
      } catch (error) {
        console.error("Error fetching scores:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchScores();

    // Subscribe to real-time updates
    const subscription = supabase
      .channel("dart_scores_channel")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "dart_scores",
        },
        (payload) => {
          const newScore = payload.new as DartScore;

          // Update recent scores
          setRecentScores((prevScores) => {
            const updatedScores = [newScore, ...prevScores].slice(0, 10);
            return updatedScores;
          });

          // Update game state
          setGameState((prevState) => {
            const newState = { ...prevState };

            if (newScore.player === "robot") {
              newState.robotScore += newScore.score;
            } else {
              newState.humanScore += newScore.score;
            }

            newState.currentRound = Math.max(
              prevState.currentRound,
              newScore.round,
            );
            newState.lastThrow = {
              player: newScore.player,
              score: newScore.score,
            };

            return newState;
          });
        },
      )
      .subscribe();

    return () => {
      supabase.removeChannel(subscription);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="h-12 w-12 animate-spin">
          <Target className="text-primary h-12 w-12" />
        </div>
      </div>
    );
  }

  // Replace the return statement with this updated version that shows setup instructions if needed
  if (!databaseReady) {
    return <DatabaseSetupGuide />;
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card className="border-4 border-blue-500">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Robot className="h-6 w-6" />
              <span>Robot</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="py-4 text-center text-6xl font-bold">
              {gameState.robotScore}
            </div>
          </CardContent>
        </Card>

        <Card className="border-4 border-green-500">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <User className="h-6 w-6" />
              <span>Human</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="py-4 text-center text-6xl font-bold">
              {gameState.humanScore}
            </div>
          </CardContent>
        </Card>
      </div>

      {gameState.lastThrow && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Last Throw</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {gameState.lastThrow.player === "robot" ? (
                  <Robot className="h-6 w-6" />
                ) : (
                  <User className="h-6 w-6" />
                )}
                <span className="text-lg font-medium capitalize">
                  {gameState.lastThrow.player}
                </span>
              </div>
              <Badge variant="outline" className="px-3 py-1 text-xl">
                +{gameState.lastThrow.score}
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Recent Throws</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {recentScores.length > 0 ? (
              recentScores.map((score, index) => (
                <div
                  key={score.id}
                  className="bg-muted/50 flex items-center justify-between rounded-md p-2"
                >
                  <div className="flex items-center gap-2">
                    {score.player === "robot" ? (
                      <Robot className="h-5 w-5" />
                    ) : (
                      <User className="h-5 w-5" />
                    )}
                    <span className="capitalize">{score.player}</span>
                    <span className="text-muted-foreground text-sm">
                      Round {score.round}
                    </span>
                  </div>
                  <Badge variant="outline">+{score.score}</Badge>
                </div>
              ))
            ) : (
              <div className="text-muted-foreground py-4 text-center">
                No scores yet
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
