import type { Metadata } from "next";
import { Rajdhani } from "next/font/google";
import "./globals.css";

// Components
import Navbar from "./components/navbar";
import { AuthProvider } from "./context/AuthContext"; // TEMP AUTH

const rajdhaniSans = Rajdhani({
  subsets: ["latin"],
  variable: "--font-rajdhani",
  display: "swap",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "ONC AI Assistant",
  description: "ONC AI Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={rajdhaniSans.variable}>
        <AuthProvider>
          {" "}
          {/* TEMP AUTH */}
          <Navbar />
          {children}
        </AuthProvider>{" "}
        {/* TEMP AUTH */}
      </body>
    </html>
  );
}
