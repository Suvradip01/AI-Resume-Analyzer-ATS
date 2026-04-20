import { SignUp } from "@clerk/clerk-react";

export default function SignUpPage() {
    return (
        <div className="flex items-center justify-center min-h-screen bg-background text-foreground">
            <SignUp path="/sign-up" routing="path" signInUrl="/sign-in" />
        </div>
    );
}
