function integral = sef(y, eta)
    % Scaled Error Function (SEF)
    sign_y = sign(y);  
    y_abs = abs(y);    
    num_points = 1000; 
    z = linspace(0, max(y_abs(:)), num_points); 
    integrand = exp(-z.^2 / (2 * eta)); 
    integral_table = cumtrapz(z, integrand); 
    integral_abs = interp1(z, integral_table, y_abs, 'linear', 'extrap');
    integral = sign_y .* integral_abs;
end