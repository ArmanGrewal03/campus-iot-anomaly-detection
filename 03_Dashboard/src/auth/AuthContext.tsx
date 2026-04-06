import * as React from 'react';

export type AuthRole = 'guest' | 'admin';

type AuthUser = {
  role: AuthRole;
  displayName: string;
  email: string;
};

type AuthContextValue = {
  user: AuthUser;
  isAdmin: boolean;
  login: (username: string, password: string) => { success: boolean; message?: string };
  logoutToGuest: () => void;
};

const USER_BY_ROLE: Record<AuthRole, AuthUser> = {
  guest: {
    role: 'guest',
    displayName: 'C_IOT GUEST',
    email: 'guest@campusiot.local',
  },
  admin: {
    role: 'admin',
    displayName: 'C_IOT ADMIN',
    email: 'admin@campusiot.local',
  },
};

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [role, setRole] = React.useState<AuthRole>('guest');

  const login = React.useCallback((username: string, password: string) => {
    const trimmedUser = username.trim().toLowerCase();
    const trimmedPass = password.trim().toLowerCase();

    if (trimmedUser === 'guest' && trimmedPass === 'guest') {
      setRole('guest');
      return { success: true };
    }

    if (trimmedUser === 'admin' && trimmedPass === 'admin') {
      setRole('admin');
      return { success: true };
    }

    return { success: false, message: 'Invalid username or password.' };
  }, []);

  const logoutToGuest = React.useCallback(() => {
    setRole('guest');
  }, []);

  const value = React.useMemo<AuthContextValue>(() => ({
    user: USER_BY_ROLE[role],
    isAdmin: role === 'admin',
    login,
    logoutToGuest,
  }), [role, login, logoutToGuest]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = React.useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}
