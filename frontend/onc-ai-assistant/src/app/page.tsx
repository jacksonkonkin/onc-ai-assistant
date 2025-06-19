"use client";
import './landingPage.css'; // global CSS import
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function Home() {
  const router = useRouter();
  const goToLogIn = () => {
    router.push("/authentication");
  };
  const goToSignUp = () => {
    router.push("/authentication/signUp");
  };

  return (
    <div className="landing-container">
      <Image
        src="/landingPageBG.jpg"
        alt="Background"
        layout="fill"
        objectFit="cover"
        className="bg-image"
      />

      <main className="landing-hero">
        <h1>Meet Ocean Networks Canada’s Virtual Assistant!</h1>
        <p>
          Ocean Networks Canada’s AI-powered virtual assistant is here to help you navigate their{' '}
          <a href="https://data.oceannetworks.ca/">Oceans 3.0 Data Portal</a> and discover our oceans.
        </p>
        <p>Log in or create an account to get started!</p>
        <div className="landing-buttons">
          <button onClick={goToLogIn} className="landing-ctaBtn">Log In</button>
          <button onClick={goToSignUp} className="landing-ctaBtn">Sign Up</button>
        </div>

        {/* Territorial Acknowledgment */}
        <p className="territorial-ack">
          We acknowledge with respect that the Cambridge Bay coastal community observatory is located on the lands and in the waters of the Inuit, in Iqaluktuuttiaq (Cambridge Bay) in the Kitikmeot Region of Nunavut.
        </p>
      </main>
    </div>
  );
}
