import Link from "next/link";
import { Terminal, Box, Settings, Cpu, Mic, LayoutGrid, Github } from "lucide-react";

export default function Home() {
    return (
        <main className="flex min-h-screen flex-col items-center p-8 relative overflow-hidden">
            {/* Background Gradient */}
            <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-primary/20 to-transparent pointer-events-none" />

            {/* Header */}
            <nav className="w-full max-w-6xl flex justify-between items-center z-10 mb-20">
                <div className="text-2xl font-bold flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary rounded-sm" />
                    UNLZ AI STUDIO
                </div>
                <div className="flex gap-6 text-sm font-medium text-gray-400">
                    <Link href="#" className="hover:text-primary transition-colors">Documentation</Link>
                    <Link href="#" className="hover:text-primary transition-colors">GitHub</Link>
                    <button className="text-white bg-white/10 px-4 py-2 rounded-full hover:bg-white/20 transition-colors">
                        Beta v2.0
                    </button>
                </div>
            </nav>

            {/* Hero */}
            <div className="z-10 text-center max-w-3xl mb-24">
                <h1 className="text-6xl md:text-8xl font-bold tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-b from-white to-white/50">
                    BUILD WITH <br />
                    <span className="text-primary">INTELLIGENCE</span>
                </h1>
                <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
                    The modular AI platform related to UNLZ engineering.
                    Deploy LLMs, process 3D Splats, and ensure accessibility.
                </p>
                <div className="flex justify-center gap-4">
                    <Link href="/modules" className="bg-primary text-white px-8 py-3 rounded-md font-bold hover:bg-primary/80 transition-colors flex items-center gap-2">
                        <LayoutGrid size={20} />
                        Explore Modules
                    </Link>
                    <Link href="/monitor" className="border border-white/20 text-white px-8 py-3 rounded-md font-bold hover:bg-white/10 transition-colors flex items-center gap-2">
                        <Cpu size={20} />
                        System Monitor
                    </Link>
                </div>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-6xl z-10">
                <FeatureCard
                    icon={<Box className="text-primary" size={32} />}
                    title="Gaussian Splatting"
                    desc="Create and view 3D scenes from standard images using SharpSplat technology."
                />
                <FeatureCard
                    icon={<Terminal className="text-green-500" size={32} />}
                    title="LLM Manager"
                    desc="Local Llama server management. Chat, download GGUFs, and tune parameters."
                />
                <FeatureCard
                    icon={<Mic className="text-primary" size={32} />}
                    title="Inclu-IA"
                    desc="Real-time subtitling server for accessible classrooms. Adapted from RPi."
                />
            </div>

            <footer className="mt-20 text-gray-600 text-sm">
                Factory of the Future © 2026 UNLZ
            </footer>
        </main>
    );
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
    return (
        <div className="bg-white/5 border border-white/10 p-6 rounded-xl hover:bg-white/10 transition-all cursor-default group">
            <div className="mb-4 bg-white/5 w-16 h-16 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
                {icon}
            </div>
            <h3 className="text-xl font-bold mb-2 text-white">{title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
        </div>
    );
}
