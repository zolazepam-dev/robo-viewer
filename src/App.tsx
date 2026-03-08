import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { APITester } from "./APITester";
import { Visualizer3D } from "./Visualizer3D";
import "./index.css";

export function App() {
  const [telemetry, setTelemetry] = useState<any>(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8080/telemetry");
        const data = await res.json();
        setTelemetry(data);
      } catch (e) {
        // Silently fail if server is down
      }
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container mx-auto p-8 flex flex-col gap-8">
      <Card className="bg-zinc-950 border-zinc-800 text-zinc-100">
        <CardHeader>
          <CardTitle className="text-3xl font-bold flex justify-between items-center">
            JOLTrl Go-Native 3D Training
            <span className="text-sm font-mono text-zinc-500">
              SPS: {telemetry?.sps?.toFixed(2) || "0.00"} | Step: {telemetry?.step || 0}
            </span>
          </CardTitle>
          <CardDescription className="text-zinc-400">
            Real-time 3D visualization of the parallel Go physics environment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Visualizer3D robotStates={telemetry?.robot_states || []} />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card className="bg-zinc-950 border-zinc-800 text-zinc-100">
          <CardHeader>
            <CardTitle>Telemetry Stream</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs bg-black p-4 rounded overflow-auto h-[200px]">
              {JSON.stringify(telemetry, null, 2)}
            </pre>
          </CardContent>
        </Card>
        
        <Card className="bg-zinc-950 border-zinc-800 text-zinc-100">
          <CardHeader>
            <CardTitle>API Debugger</CardTitle>
          </CardHeader>
          <CardContent>
            <APITester />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default App;
