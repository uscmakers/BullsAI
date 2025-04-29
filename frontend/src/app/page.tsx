import { Scoreboard } from "@/components/scoreboard";

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8 text-center">
        {/* <div className="mb-4 flex justify-end">
          <Button asChild variant="outline" size="sm">
            <Link href="/test">Test Panel</Link>
          </Button>
        </div> */}
        <h1 className="mb-2 text-4xl font-bold">Dart Robot Scoreboard</h1>
        {/* <p className="text-muted-foreground mx-auto max-w-2xl">
          Real-time score tracking for robot vs. human dart matches. This
          application uses Supabase Realtime to display scores as they're
          detected by your computer vision system.
        </p> */}
      </header>

      <main>
        <Scoreboard />
      </main>
    </div>
  );
}
