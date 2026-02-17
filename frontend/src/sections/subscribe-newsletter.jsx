import SectionTitle from "../components/section-title";
import { motion } from "framer-motion";
export default function SubscribeNewsletter() {
    return (
        <section className="flex flex-col items-center">
            <SectionTitle title="Join our community" description="Stay updated with the latest career tips, resume hacks, and industry trends." />
            <motion.div className="flex items-center justify-center mt-10 border border-slate-700 focus-within:outline focus-within:outline-primary text-sm rounded-full h-14 max-w-xl w-full"
                initial={{ y: 150, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 320, damping: 70, mass: 1 }}
            >
                <input type="text" className="bg-transparent outline-none rounded-full px-4 h-full flex-1 placeholder:text-slate-400" placeholder="Enter your email address" />
                <button className="bg-primary text-black rounded-full h-11 mr-1 px-10 flex items-center justify-center hover:bg-neutral-200 active:scale-95 transition">
                    Subscribe
                </button>
            </motion.div>
        </section>
    );
}