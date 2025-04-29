"use client";

import type React from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { supabase } from "@/lib/supabase";
import { Loader2 } from "lucide-react";
import { useState } from "react";

export function TestScoreForm() {
  const [player, setPlayer] = useState<"robot" | "human">("robot");
  const [score, setScore] = useState("0");
  const [round, setRound] = useState("1");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccess(false);

    try {
      const { error } = await supabase.from("dart_scores").insert([
        {
          player,
          score: Number.parseInt(score),
          round: Number.parseInt(round),
        },
      ]);

      if (error) throw error;

      setSuccess(true);
      // Reset form
      setScore("0");
    } catch (err: any) {
      setError(err.message || "Failed to add score");
    } finally {
      setIsSubmitting(false);
      // Clear success message after 3 seconds
      if (success) {
        setTimeout(() => setSuccess(false), 3000);
      }
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Test Score Entry</CardTitle>
        <CardDescription>Add scores manually for testing</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label>Player</Label>
            <RadioGroup
              value={player}
              onValueChange={(value) => setPlayer(value as "robot" | "human")}
              className="flex space-x-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="robot" id="robot" />
                <Label htmlFor="robot">Robot</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="human" id="human" />
                <Label htmlFor="human">Human</Label>
              </div>
            </RadioGroup>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="score">Score</Label>
              <Input
                id="score"
                type="number"
                min="0"
                max="180"
                value={score}
                onChange={(e) => setScore(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="round">Round</Label>
              <Input
                id="round"
                type="number"
                min="1"
                value={round}
                onChange={(e) => setRound(e.target.value)}
                required
              />
            </div>
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}
          {success && (
            <p className="text-sm text-green-500">Score added successfully!</p>
          )}
        </form>
      </CardContent>
      <CardFooter>
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="w-full"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Adding Score
            </>
          ) : (
            "Add Score"
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}
