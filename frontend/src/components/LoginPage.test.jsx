import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import LoginPage from "./LoginPage.jsx";

describe("LoginPage", () => {
  it("shows the project title, the team and no fake extras", () => {
    render(<LoginPage onLogin={() => {}} login={vi.fn()} />);
    expect(screen.getByRole("heading", { name: /Crowd Safety\s*Command Centre/ })).toBeInTheDocument();
    expect(screen.getByText("Nashik Kumbh Mela 2027")).toBeInTheDocument();
    expect(screen.getByText(/Sanjivani College of Engineering, Kopargaon/)).toBeInTheDocument();
    expect(screen.queryByText(/forgot/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/sign up|register/i)).not.toBeInTheDocument();
  });

  it("asks for both fields before calling the backend", async () => {
    const login = vi.fn();
    render(<LoginPage onLogin={() => {}} login={login} />);
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Enter your username and password.");
    expect(login).not.toHaveBeenCalled();
  });

  it("shows and hides the password", async () => {
    render(<LoginPage onLogin={() => {}} login={vi.fn()} />);
    const field = screen.getByLabelText("Password");
    expect(field).toHaveAttribute("type", "password");
    await userEvent.click(screen.getByRole("button", { name: "Show password" }));
    expect(field).toHaveAttribute("type", "text");
    expect(screen.getByRole("button", { name: "Hide password" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the loading state, then the backend's reason on a wrong password", async () => {
    let reject;
    const login = vi.fn(() => new Promise((_, r) => { reject = r; }));
    render(<LoginPage onLogin={() => {}} login={login} />);
    await userEvent.type(screen.getByLabelText("Username"), "officer");
    await userEvent.type(screen.getByLabelText("Password"), "wrong-password");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(screen.getByRole("button", { name: "Signing in…" })).toBeDisabled();
    reject(new Error("Wrong username or password."));
    expect(await screen.findByRole("alert")).toHaveTextContent("Wrong username or password.");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
    expect(login).toHaveBeenCalledWith("officer", "wrong-password");
  });

  it("hands the session on after a successful login, and shows why the last one ended", async () => {
    const value = { token: "t", expires_at: 2e9, user: { username: "admin", name: "Admin", role: "admin" } };
    const onLogin = vi.fn();
    render(<LoginPage onLogin={onLogin} login={vi.fn().mockResolvedValue(value)} notice="Your session has expired: please log in again." />);
    expect(screen.getByRole("status")).toHaveTextContent("Your session has expired");
    await userEvent.type(screen.getByLabelText("Username"), " admin ");
    await userEvent.type(screen.getByLabelText("Password"), "long-enough-pw");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(onLogin).toHaveBeenCalledWith(value);
  });
});
